# Dataset README

This project uses HAM10000 with metadata + original image files.

## Expected structure

```text
dataset/
├── metadata/
│   └── HAM10000_metadata.csv
├── images_part_1/
├── images_part_2/
└── images_sample/
```

## Notes

- `images_part_1` and `images_part_2` are the main image sources.
- `images_sample` is for quick local testing/smoke checks.
- The code maps image paths using `image_id` from metadata.
- No physical `train/val/test` directories are needed.
- Keep large dataset images out of Git.
