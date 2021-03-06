```python
import ipywidgets as wg
from PIL import Image

def f(x):
    if x == 1:
        return Image.open("images/cartoon/original.jpg")
    elif x == 2:
        return Image.open("images/cartoon/x2_scaled.jpg")
    elif x == 3:
        return Image.open("images/cartoon/x4_scaled.jpg")
    else:
        return Image.open("images/cartoon/x8_scaled.jpg")

wg.interact(f, x=wg.IntSlider(min=1,max=4,step=1));
```


    interactive(children=(IntSlider(value=1, description='x', max=4, min=1), Output()), _dom_classes=('widget-inte…



```python

```
