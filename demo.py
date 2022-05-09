
import argparse
import os
from run_model import text_to_image, load_sentence_model, mapping_model as mp

def demo(args):
    """Demo text_to_image from a text file with sentences."""

    demo_path = args.demo_path
    images_path = args.image_path

    # Read sample sentences:

    with open(demo_path, "r", encoding="utf-8") as f:
        texts = f.read().splitlines()

    # Load fine-tuned sequence classification model:
    model, tokenizer = load_sentence_model()

    # Image generation from our sample sentences:

    for i, text in enumerate(texts):
        # verbose prints the predicted ImageNet class from our sentence
        img = text_to_image(text, mapping_model=mp, lm_model=model, lm_tokenizer=tokenizer, gan_model=None, verbose=1)
        img.save(os.path.join(images_path, f"{i}.png"))

    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--demo_path", default="./demo_texts.txt", type=str,
                        help="Path to text file with demo sentences.")
    parser.add_argument("--image_path", default="./sample_images/", type=str,
                        help="Folder to save generated sample images.")


    args = parser.parse_args()
    demo(args)


