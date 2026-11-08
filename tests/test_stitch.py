import unittest

from PIL import Image

from mistral_stitch import stitch_images


class StitchImagesTests(unittest.TestCase):
    def test_stitches_all_images_in_order_at_first_image_height(self):
        images = [
            Image.new("RGB", (80, 40), "red"),
            Image.new("RGB", (20, 20), "green"),
            Image.new("RGB", (30, 60), "blue"),
        ]

        stitched = stitch_images(images)

        self.assertEqual(len(stitched), 1)
        self.assertEqual(stitched[0].size, (140, 40))
        self.assertEqual(stitched[0].getpixel((0, 20)), (255, 0, 0))
        self.assertEqual(stitched[0].getpixel((80, 20)), (0, 128, 0))
        self.assertEqual(stitched[0].getpixel((139, 20)), (0, 0, 255))
        self.assertEqual([image.size for image in images], [(80, 40), (20, 20), (30, 60)])

    def test_zero_or_one_image_needs_no_stitching(self):
        image = Image.new("RGB", (10, 20))
        self.assertEqual(stitch_images([]), [])
        self.assertIs(stitch_images([image])[0], image)


if __name__ == "__main__":
    unittest.main()
