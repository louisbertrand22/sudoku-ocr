def main():
    import argparse
    from .pipeline import run
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True)
    parser.add_argument('--out', default='data/outputs/result.jpg')
    parser.add_argument('--backend', choices=['cnn','tesseract'], default='cnn')
    args = parser.parse_args()
    run(args.image, args.out, {'ocr': {'backend': args.backend}})

if __name__ == '__main__':
    main()
