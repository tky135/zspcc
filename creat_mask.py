
import torch
import torchvision
image_name = "test.png"

# read image as tensor
image = torchvision.io.read_image(image_name)

mask = image > 50
mask = mask * 255
mask = mask.type(torch.uint8)

# save images
torchvision.io.write_png(image, "image.png")
torchvision.io.write_png(mask, "mask.png")


