using System;
using System.Diagnostics;
using System.IO;
using System.IO.Pipes;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace Candle.GptOss {
    public sealed class ProbeResult {
        public int Pid;
        public string CreationUtc, Executable, Stdout, Stderr;
        public int ExitCode;
        public double ElapsedMs;
    }

    // Both pipes are drained concurrently into capped buffers. Every wait,
    // including pipe completion and native Job teardown, has a finite bound.
    public static class ObserverProbe {
        [StructLayout(LayoutKind.Sequential)] struct BasicLimits {
            public long ProcessTime, JobTime;
            public uint Flags;
            public UIntPtr MinimumWorkingSet, MaximumWorkingSet;
            public uint ActiveProcessLimit;
            public UIntPtr Affinity;
            public uint PriorityClass, SchedulingClass;
        }
        [StructLayout(LayoutKind.Sequential)] struct IoCounters {
            public ulong ReadOperations, WriteOperations, OtherOperations,
                ReadBytes, WriteBytes, OtherBytes;
        }
        [StructLayout(LayoutKind.Sequential)] struct ExtendedLimits {
            public BasicLimits Basic;
            public IoCounters Io;
            public UIntPtr ProcessMemory, JobMemory, PeakProcessMemory, PeakJobMemory;
        }
        [StructLayout(LayoutKind.Sequential)] struct Accounting {
            public long TotalUser, TotalKernel, PeriodUser, PeriodKernel;
            public uint PageFaults, TotalProcesses, ActiveProcesses, TerminatedProcesses;
        }
        [DllImport("kernel32.dll", CharSet=CharSet.Unicode, SetLastError=true)]
        static extern IntPtr CreateJobObject(IntPtr attributes, string name);
        [DllImport("kernel32.dll", SetLastError=true)]
        static extern bool SetInformationJobObject(IntPtr job, int kind, ref ExtendedLimits limits, uint size);
        [DllImport("kernel32.dll", SetLastError=true)]
        static extern bool AssignProcessToJobObject(IntPtr job, IntPtr process);
        [DllImport("kernel32.dll", SetLastError=true)]
        static extern bool TerminateJobObject(IntPtr job, uint exitCode);
        [DllImport("kernel32.dll", SetLastError=true)]
        static extern bool QueryInformationJobObject(IntPtr job, int kind, out Accounting accounting, uint size, IntPtr returned);
        [DllImport("kernel32.dll")] static extern bool CloseHandle(IntPtr handle);
        [StructLayout(LayoutKind.Sequential, CharSet=CharSet.Unicode)] struct Startup {
            public int Size; public string Reserved, Desktop, Title;
            public int X, Y, XSize, YSize, XCount, YCount, Fill, Flags;
            public short Show, ReservedSize; public IntPtr ReservedBytes, Input, Output, Error;
        }
        [StructLayout(LayoutKind.Sequential)] struct NativeProcess {
            public IntPtr Process, Thread; public uint Pid, Tid;
        }
        [DllImport("kernel32.dll", CharSet=CharSet.Unicode, SetLastError=true)]
        static extern bool CreateProcess(string application, StringBuilder command, IntPtr processAttributes,
            IntPtr threadAttributes, bool inherit, uint flags, IntPtr environment, string directory,
            ref Startup startup, out NativeProcess process);
        [DllImport("kernel32.dll", SetLastError=true)] static extern uint ResumeThread(IntPtr thread);
        [DllImport("kernel32.dll", SetLastError=true)] static extern bool TerminateProcess(IntPtr process, uint code);

        static string Quote(string value) {
            var result = new StringBuilder("\""); int slashes = 0;
            foreach (char c in value) {
                if (c == '\\') { slashes++; continue; }
                if (c == '"') { result.Append('\\', 2 * slashes + 1); result.Append(c); }
                else { result.Append('\\', slashes); result.Append(c); }
                slashes = 0;
            }
            result.Append('\\', 2 * slashes); return result.Append('"').ToString();
        }

        static async Task<string> Capture(Stream stream, int limit, CancellationToken cancel) {
            using (var buffer = new MemoryStream()) {
                var chunk = new byte[1024];
                while (true) {
                    int count = await stream.ReadAsync(chunk, 0, chunk.Length, cancel).ConfigureAwait(false);
                    if (count == 0) break;
                    if (buffer.Length + count > limit) throw new InvalidDataException("Observer output exceeds its byte bound");
                    buffer.Write(chunk, 0, count);
                }
                return new UTF8Encoding(false, true).GetString(buffer.ToArray());
            }
        }

        public static ProbeResult Run(string executable, string[] arguments, int timeoutMs, int byteLimit) {
            if (timeoutMs < 50 || timeoutMs > 10000 || byteLimit < 1 || byteLimit > 131072)
                throw new ArgumentOutOfRangeException("Observer bounds are invalid");
            IntPtr job = CreateJobObject(IntPtr.Zero, null);
            if (job == IntPtr.Zero) throw new InvalidOperationException("Observer Job creation failed");
            var limits = new ExtendedLimits();
            limits.Basic.Flags = 0x2000 | 0x200; // kill-on-close, Job memory
            limits.JobMemory = (UIntPtr)(256UL * 1024 * 1024);
            bool assigned = false, started = false;
            Process process = null;
            NativeProcess native = new NativeProcess();
            using (var outputPipe = new AnonymousPipeServerStream(PipeDirection.In, HandleInheritability.Inheritable))
            using (var errorPipe = new AnonymousPipeServerStream(PipeDirection.In, HandleInheritability.Inheritable))
            using (var inputPipe = new AnonymousPipeServerStream(PipeDirection.Out, HandleInheritability.Inheritable))
            using (var cancel = new CancellationTokenSource()) {
                try {
                    if (!SetInformationJobObject(job, 9, ref limits, (uint)Marshal.SizeOf<ExtendedLimits>()))
                        throw new InvalidOperationException("Observer Job limits failed");
                    var command = new StringBuilder(Quote(executable));
                    foreach (string argument in arguments) command.Append(' ').Append(Quote(argument));
                    var startup = new Startup { Size=Marshal.SizeOf<Startup>(), Flags=0x100,
                        Input=inputPipe.ClientSafePipeHandle.DangerousGetHandle(),
                        Output=outputPipe.ClientSafePipeHandle.DangerousGetHandle(),
                        Error=errorPipe.ClientSafePipeHandle.DangerousGetHandle() };
                    var timer = Stopwatch.StartNew();
                    started = CreateProcess(executable, command, IntPtr.Zero, IntPtr.Zero, true,
                        0x08000004, IntPtr.Zero, null, ref startup, out native); // no window, suspended
                    if (!started) throw new InvalidOperationException("Observer process did not start");
                    assigned = AssignProcessToJobObject(job, native.Process);
                    if (!assigned) throw new InvalidOperationException("Observer Job assignment failed");
                    process = Process.GetProcessById((int)native.Pid);
                    outputPipe.DisposeLocalCopyOfClientHandle(); errorPipe.DisposeLocalCopyOfClientHandle();
                    inputPipe.DisposeLocalCopyOfClientHandle(); inputPipe.Dispose();
                    if (ResumeThread(native.Thread) == uint.MaxValue) throw new InvalidOperationException("Observer resume failed");
                    var result = new ProbeResult { Pid=process.Id,
                        CreationUtc=process.StartTime.ToUniversalTime().ToString("o"), Executable=executable };
                    cancel.CancelAfter(timeoutMs);
                    var stdout = Capture(outputPipe, byteLimit, cancel.Token);
                    var stderr = Capture(errorPipe, byteLimit, cancel.Token);
                    var exit = process.WaitForExitAsync(cancel.Token);
                    // Do not use synchronous ReadToEnd or an infinite WaitForExit.
                    var all = Task.WhenAll(stdout, stderr, exit);
                    while (!all.IsCompleted) {
                        if (stdout.IsFaulted || stderr.IsFaulted) throw new InvalidDataException("Observer output collection failed or exceeds its byte bound");
                        if (timer.ElapsedMilliseconds >= timeoutMs) throw new TimeoutException("Observer probe deadline expired");
                        Thread.Sleep(10);
                    }
                    all.GetAwaiter().GetResult();
                    result.ExitCode=process.ExitCode; result.Stdout=stdout.Result; result.Stderr=stderr.Result;
                    result.ElapsedMs=timer.Elapsed.TotalMilliseconds;
                    return result;
                } finally {
                    cancel.Cancel();
                    try {
                        if (assigned) {
                            if (!TerminateJobObject(job, 1)) throw new InvalidOperationException("Observer Job termination failed");
                            var cleanup = Stopwatch.StartNew();
                            while (true) {
                                Accounting state;
                                if (!QueryInformationJobObject(job, 1, out state, (uint)Marshal.SizeOf<Accounting>(), IntPtr.Zero))
                                    throw new InvalidOperationException("Observer Job cleanup query failed");
                                if (state.ActiveProcesses == 0) break;
                                if (cleanup.ElapsedMilliseconds >= 2000) throw new TimeoutException("Observer Job tree remains after cleanup");
                                Thread.Sleep(10);
                            }
                        } else if (started) {
                            if (!TerminateProcess(native.Process, 1)) throw new InvalidOperationException("Suspended observer termination failed");
                        }
                        if (process != null && !process.WaitForExit(2000)) throw new TimeoutException("Observer process remains after cleanup");
                    } finally {
                        if (process != null) process.Dispose();
                        if (native.Thread != IntPtr.Zero) CloseHandle(native.Thread);
                        if (native.Process != IntPtr.Zero) CloseHandle(native.Process);
                        CloseHandle(job);
                    }
                }
            }
        }
    }
}
