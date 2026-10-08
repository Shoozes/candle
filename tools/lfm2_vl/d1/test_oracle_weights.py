"""Small retained-operand reader tests; no model load or forward."""
import io
import struct
import unittest

import numpy as np

from oracle_weights import decode, directory, text_name, vision_name


def gguf(entries, payload):
    def string(value):
        value = value.encode()
        return struct.pack("<Q", len(value)) + value
    header = bytearray(b"GGUF" + struct.pack("<IQQ", 3, len(entries), 0))
    for name, shape, kind, offset in entries:
        header += string(name) + struct.pack("<I", len(shape))
        header += struct.pack("<" + "Q" * len(shape), *reversed(shape))
        header += struct.pack("<IQ", kind, offset)
    header += bytes((-len(header)) % 32)
    return bytes(header) + payload


class Operands(unittest.TestCase):
    def test_q8_signed_coefficients_and_original_float_values(self):
        codes = np.arange(-16, 16, dtype=np.int8)
        payload = struct.pack("<e", 0.5) + codes.tobytes()
        payload += struct.pack("<2e", -1.25, 3.5)
        payload += struct.pack("<2f", -0.125, 17.0)
        source = gguf([("q", (1, 32), 8, 0), ("h", (2,), 1, 34),
                       ("f", (2,), 0, 38)], payload)
        tensors = directory(io.BytesIO(source))
        np.testing.assert_array_equal(decode(source, tensors["q"]), (codes * 0.5).reshape(1, 32))
        np.testing.assert_array_equal(decode(source, tensors["h"]), [-1.25, 3.5])
        np.testing.assert_array_equal(decode(source, tensors["f"]), [-0.125, 17.0])

    def test_directory_rejects_overlap_truncation_and_incomplete_q8(self):
        cases = [gguf([("a", (2,), 0, 0), ("b", (2,), 0, 4)], bytes(16)),
                 gguf([("a", (2,), 0, 0)], bytes(7)),
                 gguf([("a", (31,), 8, 0)], bytes(34))]
        for source in cases:
            with self.subTest(length=len(source)), self.assertRaises(ValueError):
                directory(io.BytesIO(source))

    def test_directory_rejects_duplicate_and_unapproved_dtype(self):
        for entries in [[("a", (1,), 0, 0), ("a", (1,), 0, 4)], [("a", (32,), 2, 0)]]:
            with self.assertRaises(ValueError):
                directory(io.BytesIO(gguf(entries, bytes(64))))

    def test_pinned_flat_graph_names_and_unmapped_names(self):
        self.assertEqual(text_name("token_embd_norm.weight"), "model.language_model.embedding_norm.weight")
        self.assertEqual(text_name("blk.0.shortconv.conv.weight"), "model.language_model.layers.0.conv.conv.weight")
        self.assertEqual(vision_name("v.patch_embd.weight"), "model.vision_tower.embeddings.patch_embedding.weight")
        self.assertEqual(vision_name("mm.2.weight"), "model.multi_modal_projector.linear_2.weight")
        for mapping in [text_name, vision_name]:
            with self.assertRaises(ValueError):
                mapping("unmapped.weight")


if __name__ == "__main__":
    unittest.main()
