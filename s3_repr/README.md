Our project aims to perform 5-class image segmentation on [drone imagery](https://www.kaggle.com/datasets/santurini/semantic-segmentation-drone-dataset/data), using a semantic segmentation mask to identify the precise location of obstacles, water, soft-surfaces, moving-objects, and landing-zones.

We train a Computer Vision model on this segmentation dataset to detect five different classes of objects from drone imagery of urban scenes.

The trained model should enhance the safety of autonomous drone flights and landings in urban areas by distinguishing different kinds of obstacles from landing-zones.

We expect to use a CNN for the image classification and a U-net for the segmentation. We will implement our models in the pytorch library, potentially leveraging transfer-learning for classification. A U-Net architecture is chosen for the project because it performs well with small objects, preserves spacial detail via skip connections, and performs well with low-data availability

See below for instructions on how to run.

To download and preprocess the Kaggle Dataset, run

```
 # TODO
```

Dataset structure

After downloading, you will find the data is structured in the following way:

```
 # TODO add tree for raw/ classes/dataset/ label_images_semantic/,
original_images/
```

- `original_images` contains 400 png drone images, with format `<id>.png`, in RGB coloring
- `label_images_semantic/` contains the same 400 images, but pixel RGB values have been replaced with the corresponding class RGB values according to the table below:

| Class          | Color                                                                                                                         | R   | G   | B   |
| -------------- | ----------------------------------------------------------------------------------------------------------------------------- | --- | --- | --- |
| obstacles      | <span style="display:inline-block;width:16px;height:16px;background-color:rgb(155,38,182);border-radius:3px;"></span> Magenta | 155 | 38  | 182 |
| water          | <span style="display:inline-block;width:16px;height:16px;background-color:rgb(14,135,204);border-radius:3px;"></span> Blue    | 14  | 135 | 204 |
| soft-surfaces  | <span style="display:inline-block;width:16px;height:16px;background-color:rgb(124,252,0);border-radius:3px;"></span> Green    | 124 | 252 | 0   |
| moving-objects | <span style="display:inline-block;width:16px;height:16px;background-color:rgb(255,20,147);border-radius:3px;"></span> Pink    | 255 | 20  | 147 |
| landing-zones  | <span style="display:inline-block;width:16px;height:16px;background-color:rgb(169,169,169);border-radius:3px;"></span> Grey   | 169 | 169 | 169 |

_Note_: Original imagery dataset comes from [TU Graz, IVC](https://ivc.tugraz.at/research-project/semantic-drone-dataset/).
