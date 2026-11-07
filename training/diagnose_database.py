import numpy as np

def inspect_dataset(dataset, name):
    print(f"\n--- {name} ---")

    total_images = 0
    total_boxes = 0
    empty_images = 0
    bad_images = 0
    bad_boxes = 0

    for batch in dataset:
        images = batch["images"].numpy()
        boxes = batch["bounding_boxes"]["boxes"]
        classes = batch["bounding_boxes"]["classes"]

        for i in range(images.shape[0]):
            total_images += 1

            image = images[i]

            if not np.isfinite(image).all():
                bad_images += 1

            b = boxes[i].numpy()
            c = classes[i].numpy()

            if len(b) == 0:
                empty_images += 1
                continue

            total_boxes += len(b)

            # Check for invalid coordinates.
            valid = (
                np.isfinite(b).all(axis=1)
                & (b[:, 0] >= 0)
                & (b[:, 1] >= 0)
                & (b[:, 2] <= IMAGE_SIZE[1])
                & (b[:, 3] <= IMAGE_SIZE[0])
                & (b[:, 2] > b[:, 0])
                & (b[:, 3] > b[:, 1])
            )

            if not valid.all():
                bad_boxes += np.count_nonzero(~valid)

            if not np.isfinite(c).all():
                bad_boxes += 1

    print(f"Images:          {total_images}")
    print(f"Bounding boxes:  {total_boxes}")
    print(f"Empty images:    {empty_images}")
    print(f"Invalid images:  {bad_images}")
    print(f"Invalid boxes:   {bad_boxes}")


inspect_dataset(train_ds, "Training")
inspect_dataset(val_ds, "Validation")