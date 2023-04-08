# Image Keywording by CLIP

Code for finding the most suitable keywords for images from the given vocabulary using a [OpenAi-CLIP](https://github.com/openai/CLIP).

## Usage

```bash
$ python process.py --batch_size 128 --template 'photo for an article on the topic of <keyword>.' --top_k 10
```
Note, default template showed the best keyword results for stock photos.

## Good keywording samples
<br/>
<img style="display: block; margin: auto;" src="./images/16575567285458303589.jpg" width="200px">
<p style="text-align: left;">
    <strong>Keywords:</strong> <em>"wallpaper sample", "seamless pattern", "succulent plant", "cushion cactus", "wallpaper", "tropical pattern", "floral pattern", "wallpaper - decor", "african cactus", "hecho cactus"</em>
</p>
<br/>
<img style="display: block; margin: auto;" src="./images/16575307371217677131.jpg" width="200px">
<p style="text-align: left;">
    <strong>Keywords:</strong> <em>"autumn equinox", "autumn", "european beech", "deciduous beech", "deciduous tree", "national memorial arboretum", "beech tree", "copse", "hornbeam", "autumn collection"</em>
</p>

## Bad keywording samples
<br/>
<img style="display: block; margin: auto;" src="./images/16573571113509254472.jpg" width="200px">
<p style="text-align: left;">
    <strong>Keywords:</strong> <em>"earths crust", "satellite view", "topography", "corpus striatum", "moon surface", "topographic map", "stream - body of water", "marbled effect", "rock face", "petroglyph"</em>
</p>
<br/>
<img style="display: block; margin: auto;" src="./images/16575973069672867436.jpg" width="200px">
<p style="text-align: left;">
    <strong>Keywords:</strong> <em>"surgical glove", "protective glove", "splint", "boxing glove", "elastic therapeutic tape", "ivory - material", "geoduck", "ribbon - sewing item", "glove", "sports glove"</em>
</p>
