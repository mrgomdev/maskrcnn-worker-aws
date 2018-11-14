"""
Main server loop
"""
from aws import load_input, save_output, fetch_request
from inference import inference_then_save

from PIL import Image
import os
import tempfile


def preprocess_image(image: Image.Image) -> Image.Image:
    width = min(image.size)
    width, height = width, width

    processed_image = Image.new(mode='RGB', size=(width, height))
    processed_image.paste(im=image, box=(0, 0))
    processed_image = processed_image.resize((224, 224))

    return processed_image


def main():
    while True:
        try:
            messages = fetch_request(timeout_seconds=20)
        except KeyboardInterrupt as e:
            print(e)
            break
        except Exception as e:
            print(e)
        else:
            for message in messages:
                request_id = message.body
                try:
                    print('request_id:', request_id)
                    img = load_input(key='{}'.format(request_id))

                    temp_fd, temp_path = tempfile.mkstemp(suffix='.jpg')
                    try:
                        inference_then_save(img, save_path=temp_path)
                        print('request_id:', request_id, 'done')
                        save_output(key=request_id, file_path=temp_path)
                        message.delete()
                    finally:
                        os.close(temp_fd)
                        os.unlink(temp_path)
                except Exception as e:
                    print(e)


if __name__ == '__main__':
    main()
