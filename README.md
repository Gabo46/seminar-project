```python
import IPython
```


```python
html = """
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@splidejs/splide@latest/dist/css/themes/splide-skyblue.min.css">
<script src="https://cdn.jsdelivr.net/npm/@splidejs/splide@latest/dist/js/splide.min.js"></script>

<div class="splide">
    <div class="splide__track">
        <ul class="splide__list text-center">
            <li class="splide__slide">
                <h3>Original</h3>
                <img src="images/cartoon/original.jpg" style="margin-left:auto;margin-right:auto" />
            </li>
            <li class="splide__slide">
                <h3>Skalierungsfaktor: 2</h3>
                <img src="images/cartoon/x2_scaled.jpg" style="margin-left:auto;margin-right:auto" />
            </li>
            <li class="splide__slide">
                <h3>Skalierungsfaktor: 4</h3>
                <img src="images/cartoon/x4_scaled.jpg" style="margin-left:auto;margin-right:auto" />
            </li>
            <li class="splide__slide">
                <h3>Skalierungsfaktor: 8</h3>
                <img src="images/cartoon/x8_scaled.jpg" style="margin-left:auto;margin-right:auto" />
            </li>
            <li class="splide__slide">
                <h3>Skalierungsfaktor: 16</h3>
                <img src="images/cartoon/x16_scaled.jpg" style="margin-left:auto;margin-right:auto" />
            </li>
        </ul>
    </div>
</div>

<script>
    new Splide('.splide', {
        type: 'loop',
        autoplay: true
    }).mount();
</script>
"""

IPython.display.HTML(data=html)
```





<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@splidejs/splide@latest/dist/css/themes/splide-skyblue.min.css">
<script src="https://cdn.jsdelivr.net/npm/@splidejs/splide@latest/dist/js/splide.min.js"></script>

<div class="splide">
    <div class="splide__track">
        <ul class="splide__list text-center">
            <li class="splide__slide">
                <h3>Original</h3>
                <img src="images/cartoon/original.jpg" style="margin-left:auto;margin-right:auto" />
            </li>
            <li class="splide__slide">
                <h3>Skalierungsfaktor: 2</h3>
                <img src="images/cartoon/x2_scaled.jpg" style="margin-left:auto;margin-right:auto" />
            </li>
            <li class="splide__slide">
                <h3>Skalierungsfaktor: 4</h3>
                <img src="images/cartoon/x4_scaled.jpg" style="margin-left:auto;margin-right:auto" />
            </li>
            <li class="splide__slide">
                <h3>Skalierungsfaktor: 8</h3>
                <img src="images/cartoon/x8_scaled.jpg" style="margin-left:auto;margin-right:auto" />
            </li>
            <li class="splide__slide">
                <h3>Skalierungsfaktor: 16</h3>
                <img src="images/cartoon/x16_scaled.jpg" style="margin-left:auto;margin-right:auto" />
            </li>
        </ul>
    </div>
</div>

<script>
    new Splide('.splide', {
        type: 'loop',
        autoplay: true
    }).mount();
</script>





```python

```
