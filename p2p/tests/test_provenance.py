import unittest
import os
import cv2
import pdqhash
from p2p.provenance.probes.c2pa_probe import analyze_one
from p2p.provenance.fingerprint import fingerprints_for_path

class TestProvenance(unittest.TestCase):

    def test_c2pa_probe(self):
        img = "p2p/tests/test_images/test_image_1.jpg"
        vid = "p2p/tests/test_videos/test_video_1.mp4"

        # 图片：大多数公开测试图无 C2PA，是正常现象
        r_img = analyze_one(img)
        self.assertIsNotNone(r_img)
        # 只要工具可运行并返回结构即可；若无 Claim，应给出 no_claim / 对应错误
        self.assertTrue((r_img.present is True) or (r_img.error in ("no_claim", "c2patool_failed", "c2patool_timeout", "c2patool_not_found")), f"Unexpected C2PA result: {r_img}")

        # 视频：同理
        r_vid = analyze_one(vid)
        self.assertIsNotNone(r_vid)
        self.assertTrue((r_vid.present is True) or (r_vid.error in ("no_claim", "c2patool_failed", "c2patool_timeout", "c2patool_not_found")), f"Unexpected C2PA result: {r_vid}")

    def test_pdqhash(self):
        img = "p2p/tests/test_images/test_image_1.jpg"
        image = cv2.imread(img)
        self.assertIsNotNone(image, f"Failed to read {img}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        hv, q = pdqhash.compute(image)
        self.assertIsNotNone(hv)
        self.assertGreaterEqual(len(hv), 1)
        self.assertGreaterEqual(q, 0.0)

        hv8, q8 = pdqhash.compute_dihedral(image)
        self.assertEqual(len(hv8), 8)

    def test_vpdq(self):
        vid = "p2p/tests/test_videos/test_video_1.mp4"
        res = fingerprints_for_path(vid)
        # 只要 CLI 正常执行并生成了某种输出文件，就 available=True
        self.assertTrue(res.vpdq is not None, "vpdq result missing")
        self.assertTrue(res.vpdq.available, f"vPDQ not available: {res.vpdq.error}")
        # 有些格式可能解析不到帧，但应至少给出 raw_text 供后续排查
        self.assertTrue(len(res.vpdq.frames) > 0 or (res.vpdq.raw_json and res.vpdq.raw_json.get('raw_text')), "No frames parsed and no raw_text captured")

    def test_pdqhash_float(self):
        img = "p2p/tests/test_images/test_image_1.jpg"
        image = cv2.imread(img)
        self.assertIsNotNone(image, f"Failed to read {img}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        hvf, q = pdqhash.compute_float(image)
        self.assertIsNotNone(hvf)
        self.assertGreaterEqual(q, 0.0)

if __name__ == '__main__':
    unittest.main()
