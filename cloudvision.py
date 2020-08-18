import io
import os

# Imports the Google Cloud client library
from google.cloud import vision
from google.cloud.vision import types

sample_files = """line_0_index_0_char_G.png
line_0_index_10_char_A.png
line_0_index_11_char_m.png
line_0_index_12_char_o.png
line_0_index_13_char_u.png
line_0_index_14_char_n.png
line_0_index_15_char_t.png
line_0_index_16_char_(.png
line_0_index_17_char_R.png
line_0_index_18_char_M.png
line_0_index_19_char_).png
line_0_index_1_char_S.png
line_0_index_20_char_G.png
line_0_index_21_char_S.png
line_0_index_22_char_T.png
line_0_index_23_char_(.png
line_0_index_24_char_R.png
line_0_index_25_char_M.png
line_0_index_26_char_).png
line_0_index_2_char_T.png
line_0_index_3_char_S.png
line_0_index_4_char_u.png
line_0_index_5_char_m.png
line_0_index_6_char_m.png
line_0_index_7_char_a.png
line_0_index_8_char_r.png
line_0_index_9_char_y.png
line_1_index_0_char_S.png
line_1_index_10_char_3.png
line_1_index_11_char_..png
line_1_index_12_char_2.png
line_1_index_13_char_0.png
line_1_index_14_char_1.png
line_1_index_15_char_..png
line_1_index_16_char_4.png
line_1_index_17_char_0.png
line_1_index_1_char_=.png
line_1_index_2_char_G.png
line_1_index_3_char_S.png
line_1_index_4_char_T.png
line_1_index_5_char_0.png
line_1_index_6_char_6.png
line_1_index_7_char_%.png
line_1_index_8_char_:.png
line_1_index_9_char_2.png
line_2_index_0_char_Z.png
line_2_index_10_char_..png
line_2_index_11_char_0.png
line_2_index_12_char_0.png
line_2_index_13_char_0.png
line_2_index_14_char_..png
line_2_index_15_char_0.png
line_2_index_16_char_0.png
line_2_index_1_char_=.png
line_2_index_2_char_G.png
line_2_index_3_char_S.png
line_2_index_4_char_T.png
line_2_index_5_char_0.png
line_2_index_6_char_0.png
line_2_index_7_char_%.png
line_2_index_8_char_:.png
line_2_index_9_char_0.png
number2.png"""

class CloudVisionRecognizer:
    def __init__(self, dirPath):
        self.dirPath = dirPath

    def ocrImageInCloudVision(self):
        client = vision.ImageAnnotatorClient()

        # The name of the image file to annotate
        file_name = self.filePath

        # Loads the image into memory
        with io.open(file_name, 'rb') as image_file:
            content = image_file.read()

        image = types.Image(content=content)

        # Performs label detection on the image file
        response = client.label_detection(image=image)
        labels = response.label_annotations

        print('Labels:')
        for label in labels:
            print("Label from GCV:")
            print(label.description)

    def detectText(self, filePath):
        """Detects text in the file."""
        path = filePath
        client = vision.ImageAnnotatorClient()

        with io.open(path, 'rb') as image_file:
            content = image_file.read()

        image = vision.types.Image(content=content)

        response = client.text_detection(image=image)
        texts = response.text_annotations
        print('Texts:')

        for text in texts:
            print('\n"{}"'.format(text.description))

            vertices = (['({},{})'.format(vertex.x, vertex.y)
                        for vertex in text.bounding_poly.vertices])

            print('bounds: {}'.format(','.join(vertices)))

    def recognizeAllSamplesInTheDir(self):
        sample_fileNames = sample_files.splitlines()
        for sample_filename in sample_fileNames:
            self.detectText("samples_for_gcvision/"+sample_filename)

if __name__ == "__main__":
    filePath = "sample_T_letter.png"
    cvr = CloudVisionRecognizer("samples_for_gcvision")
    cvr.detectText("receipt.jpg")
