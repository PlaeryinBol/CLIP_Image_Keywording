# Image Keywording by CLIP

Code for finding the most suitable keywords for images from the given vocabulary using a [OpenAi-CLIP](https://github.com/openai/CLIP).

## Usage

```bash
$ python process.py --batch_size 128 --template 'photo for an article on the topic of <keyword>.' --top_k 10
```
Note, default template showed the best keyword results for stock photos.

## Examples
Hover over the image to see the keywords.

<p align="left">
    <img src="./images/16575567285458303589.jpg", width="150px", title='wallpaper sample, seamless pattern, succulent plant, cushion cactus, wallpaper'>
    <img src="./images/16575307371217677131.jpg", width="150px", title='autumn equinox, autumn, european beech, deciduous beech, deciduous tree'>
    <img src="./images/16573571113509254472.jpg", width="150px", title='earths crust, satellite view, topography, corpus striatum, moon surface'>
    <img src="./images/16575973069672867436.jpg", width="150px", title='surgical glove, protective glove, splint, boxing glove, elastic therapeutic tape'>
</p>
