If the input has the information of two images, I think the end-to-end network will somehow find the relation between their pixels and level of compatibility.
As a result, consider to speed up the training, I concat two images by their deep dimension. (3, 224, 224) X2 -> (6, 224, 224)
