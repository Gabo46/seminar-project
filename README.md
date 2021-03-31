# Auswirkungen der Image Super-Resolution auf die Bildqualität

### Gabriel Dogadov, Samy Hamdad, Muhammed Aydin, Aleksandrs Frank

## 1. Einleitung



In den letzten Jahren ist ein starker Fortschritt im Machine Learning zu verzeichnen. Ein Grund dafür sind neuronale Netzwerke, welche sich als sehr mächtig und vielfältig herausgestellt haben. Vor allem für <a href="https://de.wikipedia.org/wiki/Computer_Vision#:~:text=Im%20englischen%20Sprachraum%20wird%20ebenfalls,im%20industriellen%20Umfeld%20betont%20wird.">Computer Vision</a> sind <a href="https://de.wikipedia.org/wiki/Convolutional_Neural_Network">Convolutional Neural Networks</a> vielfältig einsetzbar. Eine vielversprechende Anwendung hierbei ist die Super-Resolution, bei der die Auflösung von Bildern (also die Anzahl der Pixel) stark vergrößert wird.

Das Vergrößern (Upscaling) von Bildern kann sehr nützlich sein. Statt hochauflösende Bilder zu erstellen, könnte man stattdessen kleine Bilder bequem mit Super-Resolution hochskalieren.

<img src="images/upscaling.svg" />

Beim Hochskalieren werden meist kleine Bilder auf größere Bilder mit deutlich mehr Pixeln abgebildet. Es reicht nicht die Pixelwerte vom Originalbild in das skalierte Bild zu kopieren. Es bleiben weitere Pixel übrig, denen man Werte zuweisen muss.

Eine herkömmliche Methode zum Füllen der Pixel bei vergrößerten Bildern ist die <a href="https://en.wikipedia.org/wiki/Bicubic_interpolation">Bikubische Interpolation</a>. Das Verfahren erzeugt aber häufig unscharfe Bilder, sodass die Bildqualität nach dessen Anwendung darunter leidet. Eine Lösung für dieses Problem soll die Image Super-Resolution liefern. Dazu wird ein neuronales Netzwerk darauf trainiert, Bilder unterschiedlicher Größen so zu vergrößern, dass die Bildqualität erhalten bleibt.

In diesem Bild sieht man ein um den Faktor 4 vergrößertes Bild. Die bikubische Interpolation liefert ein unscharfes Bild, wohingegen SRResNet und SRGAN, welche auf neuronalen Netzen basieren, deutlich bessere Ergebnisse liefern. 

<img src="images/methods.png" />

Das Bild wurde aus Ledig et al. (2017) entnommen.

Zwar ist das Verfahren noch sehr jung, trotzdem gibt es bereits kommerzielle Anbieter
(<a href="https://bigjpg.com/">Bigjpg</a>, <a href="https://letsenhance.io/">LetsEnhance.io</a>, 
 <a href="https://deepai.org/machine-learning-model/torch-srgan">DeepAI</a>), 
die versprechen unsere Bilder ohne jegliche Qualitätsverluste hoch zu skalieren. Diesem Versprechen möchten wir im Folgenden auf den Grund gehen.<br>
Neben der reinen Hochskalierung gibt es zusätzlich die Möglichkeit Noise von Bildern zu entfernen. Die Noise Cancellation kann also dafür sorgen, dass nicht nur die Bilder ohne Qualitätsverlust vergrößert werden, sondern auch die Bildqualität durch das Entfernen von Noise verbessert wird.

Das führt uns zu unserer Fragestellung, der wir auf den Grund gehen wollen: __Wie verändert sich die Bildqualität nach Anwendung der Super-Resolution?__

Statt jeden Anbieter einzeln zu testen, testen wir die Super-Resolution mit der <a href="https://github.com/idealo/image-super-resolution" target="_blank">Idealo Super-Resolution Library</a>. Wir untersuchen hierbei die Möglichkeiten und Grenzen des Verfahrens.<br>
Die <a href="https://github.com/idealo/image-super-resolution" target="_blank">Idealo Super-Resolution Library</a> bietet unterschiedliche Typen von neuronalen Netzwerken an, welche neben der Skalierung auch Noise Cancellation durchführen. Die Library bietet die Möglichkeit ein neuronales Netzwerk selbst zu trainieren oder ein bereits trainiertes zu verwenden. Die vortrainierten Netzwerke kommen mit unterschiedlichen Gewichten, wobei wir keine Gewichte speziell ausgewählt haben, sondern die Standardeinstellung übernommen haben.

Wir testen die Super-Resolution dann auf zwei verschiedene Weisen. Im ersten Experiment untersuchen wir die Bildqualität in Abhängigkeit von der Vergrößerungsstufe. Wir testen die Stufen x2, x4, x8 und x16 aus, wobei x2 bedeutet, dass sowohl Höhe als auch Breite um den Faktor 2 skaliert wird. Die tatsächliche Anzahl der Pixel wird hierbei auf das 4-fache erhöht.<br>
Zum Testen der Noise Cancellation skalieren wir die Bilder stets um den Faktor 2 (also die Anzahl der Pixel vervierfacht sich), aber vorher werden die Bilder mit JPEG komprimiert. Wir nutzen die Komprimierungsstufen 30%, 60%, 75%, 85% und 95% (die Komprimierungsstufe ist 100 - Quality-Parameter in PIL). Unsere Komprimierungsstufen sind nicht gleichmäßig verteilt, da kleine Komprimierungen oftmals kaum sichtbar sind. So können wir mehr hohe Komprimierungsstufen austesten und genauer ermitteln, ab welcher Komprimierung sich das Originalbild nicht mehr in guter Qualität wiederherstellen lässt.

Nach einigen Versuchen konnten wir beobachten, dass die Bildqualität bei großen Vergrößerungsfaktoren sowie hohen Komprimierungen sich wahrnehmbar verschlechtert und formulierten unsere Hypothesen wie folgt:

### Hypothesen

1. Bei kleinen Vergrößerungs- und Komprimierungsstufen ist ein Qualitätsunterschied kaum wahrnehmbar.

2. Bei größeren Vergrößerungs- und Komprimierungsstufen verschlechtert sich die Bildqualität spürbar.


## 2. Experimentelles Design

Für das Experiment wurden drei unterschiedliche Bilder verwendet: Ein Bild aus einem Cartoon mit kräftigen Farben, ein Gesichtsbild sowie eine Landschaft. Die drei unterschiedlichen Bildtypen kommen daher zustande, dass eventuell Unterschiede abhängig vom Typus des Bildes zustande kommen können. Zunächst wissen wir nicht mit welchen Bildern das neuronale Netz trainiert wurde. Abhängig von den Trainingsdaten können immense Unterschiede auftreten. Zudem kann das Bild selbst von Bedeutung sein, wie zum Beispiel die Farben oder die Anordnung der Pixel.

#### Unsere Bilder (verkleinert)


<div style="display:flex" class="text-center">
    <div>
        <img src="images/original_images.png">
    </div>
</div>
<div>
      <div>Bildquelle Cartoon: <a href="https://www.wallpapertip.com/wpic/obwwTm_cute-cartoon-wallpaper-backgrounds-tom-and-jerry/">https://www.wallpapertip.com/wpic/obwwTm_cute-cartoon-wallpaper-backgrounds-tom-and-jerry/</a></div>
      <div>Bildquelle Gesicht: <a href="https://www.pexels.com/photo/man-in-yellow-crew-neck-t-shirt-4001263/"> https://www.pexels.com/photo/man-in-yellow-crew-neck-t-shirt-4001263/</a>
      </div>
      <div>Bildquelle Landschaft: <a href="https://www.pexels.com/photo/pathway-in-between-of-green-grass-field-67211/">https://www.pexels.com/photo/pathway-in-between-of-green-grass-field-67211</a>
      </div>
</div>

Die Originalbilder in der Originalgröße sind im Anhang verlinkt.

Da die Super-Resolution darauf abzielt, große Bilder zu erzeugen, wollten wir möglichst große Bilder vergleichen, die aber auf einen Full-HD (1920x1080px) Bildschirm passen. Um dies zu erreichen, haben wir unsere Originalbilder mit den Größen 961x540px, 856x540px und 812x540px künstlich verkleinert und im Nachhinein versucht sie möglichst gut wiederherzustellen.<br>
Zunächst wurden unsere Originalbilder verkleinert, sodass sowohl Länge als auch Breite um die Faktoren 2, 4, 8 bzw. 16 verkleinert werden. Zusätzlich wurden die Bilder, dessen Länge und Breite halbiert wurden, mittels JPEG Komprimierung komprimiert. Anschließend wurden Bilder mit Super-Resolution und ggf. Noise Cancellation wieder hochskaliert und mit dem Originalbild mittels Maximum Likelihood Difference Scaling (MLDS) wie bei Charrier et al. (2007) verglichen.<br>
Da MLDS keine direkte Skala für die Bildqualität darstellt, gehen wir davon aus, dass das Originalbild stets die beste Bildqualität hat. Diese Annahme bewahrheitet sich im Folgenden für höhere Skalierungs- bzw. Komprimierungsstufen. 

<img src="images/flow.svg" />

### JPEG Komprimierung der Bilder mittels PIL

```python
from PIL import Image

rates = [30, 60, 75, 85, 95] # Compression rates

def save_compressed(img, directory):
    """
    Speichert das Bild in verschiedenen Komprimierungsstufen
    
    Parameter
    ---------
    img : Image
        Das Bild, welches komprimiert werden soll
    directory : str
        Pfad, in dem die komprimierten Bilder hinterlegt werden sollen
    
    """
    
    for rate in rates:
        quality = 100 - rate
        img.save("{}/{}.jpg".format(directory, rate), "JPEG", quality=quality)

img = Image.open("PATH_TO_IMG")
save_compressed(face, "PATH_TO_FOLDER_WITH_IMAGES")
```

### Image Superresolution mit der <a href="https://github.com/idealo/image-super-resolution" target="_blank">Idealo ISR Library</a>

```python
import numpy as np
from PIL import Image
from ISR.models import RDN

def scale_up(rdn, directory):
    """
    Upscaling für die Bilder in verschiedenen Stufen (x2, x4, x8, x16)
    
    Parameter
    ---------
    rdn : RDN
        Trainiertes Neuronales Netzwerk
    directory : str
        Pfad zum Ordner mit den Bildern
    
    """
    
    for i in range(1, 5):
        factor = 2**i
        current_image = np.array(Image.open("{}/x{}.jpg".format(directory, factor)))
        for j in range(i):
            current_image = rdn.predict(current_image)
        Image.fromarray(current_image).save("{}/x{}_scaled.jpg".format(directory, factor), "JPEG", quality=100)

def scale_up2(rdn, directory, rates):
    """
    Upscaling für die Bilder (x2) mit Noise-Cancellation
    
    Parameter
    ---------
    rdn : RDN
        Trainiertes Neuronales Netzwerk
    directory : str
        Pfad zum Ordner mit den Bildern
    rates : list
        Liste mit Komprimierungsstufen als Ganzzahl (0-100)
    
    """
    
    for rate in rates:
        current_image = np.array(Image.open("{}/{}.jpg".format(directory, rate)))
        current_image = rdn.predict(current_image)
        Image.fromarray(current_image).save("{}/{}_scaled.jpg".format(directory, rate), "JPEG", quality=100)
        
rdn = RDN(weights='psnr-small') # Nur Upscaling
rdn2 = RDN(weights='noise-cancel') # Upscaling + Noise cancellation
        
scale_up(rdn, "PATH_TO_FOLDER_WITH_IMAGES")
scale_up2(rdn2, "PATH_TO_FOLDER_WITH_IMAGES", rates = [30, 60, 75, 85, 95])
   
```

### Maximum Likelihood Difference Scaling

Die Bilder wurden mit MLDS verglichen: Der Experimentator bekam drei Bilder gleichzeitig angezeigt, wovon zwei Bilder oben in einer Reihe und das dritte Bild unten alleine angezeigt wurde. Der Experimentator musste entscheiden, welches der beiden oberen Bilder dem Bild unten weniger ähnlich sieht. Das MLDS Experiment mit Triaden wurde von jeder Versuchsperson fünf mal durchgeführt.

Im ersten Teilexperiment wurden die Skalierungsstufen (x2, x4, x8, x16) variiert und es wurde keine Komprimierung verwendet. Mit dem Original sind es 5 verschiedene Bilder.

<img src="images/mlds1.jpg" />

Im zweiten Teilexperiment wurden die Komprimierungsstufen (30%, 60%, 75%, 85%, 95%) variiert und die Skalierungsstufe war fest auf x2. Mit dem Originalbild sind es insgesamt sechs Bilder.

<img src="images/mlds2.jpg" />

Insgesamt hat jede Versuchsperson 450 Vergleiche durchgeführt, welche dann mit dem <a href="mlds/mlds_analysis.R">R Script</a> ausgewertet wurden.

## 3. Ergebnisse

Zunächst sieht man im Folgenden die im Experiment verwendeten Bilder für das Gesichtsbild. 
Für kleine Vergrößerungen und wenig Komprimierung ist kaum ein Unterschied zu erkennen, wohingegen die stark vergrößerten 
bzw. stark komprimierten Bilder deutlich als solche zu erkennen sind.

<img src="images/face/scaling_animated.gif" />

<img src="images/face/compression_animated.gif" />

Die Bilder mit kleiner Komprimierungsstufe sehen dem Original zwar ähnlich, sind aber deutlich heller als das Originalbild. Dies könnte an den Trainingsdaten des neuronalen Netzwerks liegen.

Einen Unterschied sieht man auch deutlich im Horizontal Cut. 
Hier sieht man die Intensität aller Pixel des Bildes, die sich auf dem Streifen auf mittlerer Höhe befinden.
Die rote/grüne/blaue Funktion stellt hierbei die jeweilige R/G/B-Intensität dar.

### Horizontal Cuts für verschiedene Skalierungsstufen (Gesicht)

<img src="images/face/horizontal_cut_scaling.png" width="85%" />

Je höher der Vergrößerungsfaktor, umso mehr unterscheiden sich die Intensitäten der Pixel vom Originalbild. Außerdem ist auffällig, dass bei höherer Komprimierung die Funktion abgerundeter aussieht, d.h. starke Schwingungen in der Intensität kaum vorhanden sind.

### Horizontal Cuts für verschiedene Komprimierungsstufen (Gesicht)

<img src="images/face/horizontal_cut_compression.png" width="85%" />

Das gleiche kann man auch für die verschiendenen Komprimierungen sagen: Je höher die Komprimierung, umso weniger ähnelt die Funktion dem Original.

### Auswertung für verschiedene Skalierungsstufen

<img src="images/res_scaling.png" />

Bis zur Skalierungsstufe 2 ist die Skala stets flach bzw. leicht negativ, was darauf hindeutet, dass die Bildqualität ähnlich zum Original ist.
Für Skalierungsstufe 4 ist die Skala für den Cartoon ebenfalls flach, für die anderen Bilder aber höher. Ob damit die Super-Resolution für Cartoons besser funktioniert können wir bei nur vier Testpersonen und einem neuronalen Netzwerk, dessen Gewichte und Trainingsdaten wir nicht kennen, nicht behaupten.

### Auswertung für verschiedene Komprimierungsstufen

<img src="images/res_comp.png" />

Bis zur Komprimierungsstufe 30% ist die Kurve auch hier ziemlich flach, aber etwas höher als bei den obigen Skalen.
Zu beachten ist hierbei, dass die Bilder hier sowohl um den Faktor 2 vergrößert (die Gesamtzahl der Pixel um den Faktor 4) wurden,
als auch komprimiert wurden. Außerdem ist erwähnenswert, dass Komprimierungen häufig bis zu einer Komprimierungsstufe 
kaum bemerkbar sind. Mit höheren Komprimierungsgraden ist aber ein Unterschied wieder deutlich erkennbar.

## 4. Diskussion

Wir sehen, dass die Skalen die Stimuli durchaus widerspiegeln und dies unsere Hypothesen stützt. Bis auf kleine Details und eine erhöhte Helligkeit, von der andere Versuchspersonen wahrscheinlich nichts wissen werden, da sie das Verfahren nicht kennen, unterscheidet sich die Qualität der Bilder, die um einen kleinen Faktor hochskaliert und ggf. komprimiert wurden, kaum von der des Originals. Dagegen weisen stark vergrößerte und zuvor stark komprimierte Bilder sehr viel Noise auf und dementsprechend niedrig ist die wahrgenommene Bildqualität dieser.

### Mögliche Probleme

Zwar hat sich die Hypothese weitesgehend bestätigt, jedoch sind hier noch einige Anmerkungen:
<ul>
    <li>Außer uns selbst gab es keine weiteren Probanden. Zunächst ist es fraglich, ob vier Probanden eine Aussage über die Allgemeinheit treffen können. Außerdem sind die Probanden voreingenommen, da ihnen die Bilder und die Hypothese im Vorhinein bekannt waren.</li>
    <li>
        Es wurden lediglich drei Bilder und wenige Vergrößerungs- und Komprimierungsstufen verwendet. Für eine allgemeine Aussage sind definitiv mehr und umfangreichere Experimente notwendig. Leider ist das Verfahren (MLDS) sehr aufwendig und skaliert schlecht mit der Anzahl an Variationen.
    </li>
    <li>
        In diesem Experiment wurden die Verfahren bis an ihre Grenzen getestet. Es ist durchaus unrealistisch, dass man Bilder um den Faktor 16 vergrößern möchte oder 95%-ige JPEG Komprimierung wiederherstellen möchte. Dazu ist das Verfahren auch nicht gedacht.
    </li>
    <li>
        Die Trainingsdaten des neuronalen Netzwerks sind nicht bekannt, aber von hoher Relevanz. Wie typisch im Machine Learning hängt das Ergebnis stark von den Trainingsdaten ab.
    </li>
</ul>


### Offene Fragen

Abschließend kann man sagen, dass Super-Resolution mit der Idealo ISR Library für unsere Bilder und für kleine Vergrößerungen und Komprimierungen gut funktioniert. Leider können wir nicht genau sagen, inwiefern das Verfahren besser ist als die bikubische Interpolation. Ein direkter Vergleich der Bildqualität zwischen der Super-Resolution und der bikubischen Interpolation ist durchaus interessant. Zudem wäre es sicherlich interessant herauszufinden, wie empfindlich das Verfahren auf die zum Training des neuronalen Netztes verwendete Daten ist. Die Trainingsdaten sind dementsprechend auch entscheidend dafür, ob die kommerziellen Anbieter ihr Versprechen, die Bilder ohne Qualitätsverlust zu vergrößern, einhalten können.

## 5. Anhang

### Unsere Bilder

<table class="table">
    <tr>
        <th style="text-align:center"></th>
        <th style="text-align:center">Cartoon</th>
        <th style="text-align:center">Gesicht</th>
        <th style="text-align:center">Landschaft</th>
    </tr>
    <tr>
        <td style="text-align:center">Original</td>
        <td style="text-align:center"><a href="images/cartoon/original.jpg">Zum Bild</a></td>
        <td style="text-align:center"><a href="images/face/original.jpg">Zum Bild</a></td>
        <td style="text-align:center"><a href="images/landscape/original.jpg">Zum Bild</a></td>
    </tr>
    <tr>
        <td style="text-align:center">Skalierungsstufe 2</td>
        <td style="text-align:center"><a href="images/cartoon/x2_scaled.jpg">Zum Bild</a></td>
        <td style="text-align:center"><a href="images/face/x2_scaled.jpg">Zum Bild</a></td>
        <td style="text-align:center"><a href="images/landscape/x2_scaled.jpg">Zum Bild</a></td>
    </tr>
    <tr>
        <td style="text-align:center">Skalierungsstufe 4</td>
        <td style="text-align:center"><a href="images/cartoon/x4_scaled.jpg">Zum Bild</a></td>
        <td style="text-align:center"><a href="images/face/x4_scaled.jpg">Zum Bild</a></td>
        <td style="text-align:center"><a href="images/landscape/x4_scaled.jpg">Zum Bild</a></td>
    </tr>
    <tr>
        <td style="text-align:center">Skalierungsstufe 8</td>
        <td style="text-align:center"><a href="images/cartoon/x8_scaled.jpg">Zum Bild</a></td>
        <td style="text-align:center"><a href="images/face/x8_scaled.jpg">Zum Bild</a></td>
        <td style="text-align:center"><a href="images/landscape/x8_scaled.jpg">Zum Bild</a></td>
    </tr>
    <tr>
        <td style="text-align:center">Skalierungsstufe 16</td>
        <td style="text-align:center"><a href="images/cartoon/x16_scaled.jpg">Zum Bild</a></td>
        <td style="text-align:center"><a href="images/face/x16_scaled.jpg">Zum Bild</a></td>
        <td style="text-align:center"><a href="images/landscape/x16_scaled.jpg">Zum Bild</a></td>
    </tr>
    <tr>
        <td style="text-align:center">30% Komprimierung</td>
        <td style="text-align:center"><a href="images/cartoon/30_scaled.jpg">Zum Bild</a></td>
        <td style="text-align:center"><a href="images/face/30_scaled.jpg">Zum Bild</a></td>
        <td style="text-align:center"><a href="images/landscape/30_scaled.jpg">Zum Bild</a></td>
    </tr>
    <tr>
        <td style="text-align:center">60% Komprimierung</td>
        <td style="text-align:center"><a href="images/cartoon/60_scaled.jpg">Zum Bild</a></td>
        <td style="text-align:center"><a href="images/face/60_scaled.jpg">Zum Bild</a></td>
        <td style="text-align:center"><a href="images/landscape/60_scaled.jpg">Zum Bild</a></td>
    </tr>
    <tr>
        <td style="text-align:center">75% Komprimierung</td>
        <td style="text-align:center"><a href="images/cartoon/75_scaled.jpg">Zum Bild</a></td>
        <td style="text-align:center"><a href="images/face/75_scaled.jpg">Zum Bild</a></td>
        <td style="text-align:center"><a href="images/landscape/75_scaled.jpg">Zum Bild</a></td>
    </tr>
    <tr>
        <td style="text-align:center">85% Komprimierung</td>
        <td style="text-align:center"><a href="images/cartoon/85_scaled.jpg">Zum Bild</a></td>
        <td style="text-align:center"><a href="images/face/85_scaled.jpg">Zum Bild</a></td>
        <td style="text-align:center"><a href="images/landscape/85_scaled.jpg">Zum Bild</a></td>
    </tr>
    <tr>
        <td style="text-align:center">95% Komprimierung</td>
        <td style="text-align:center"><a href="images/cartoon/95_scaled.jpg">Zum Bild</a></td>
        <td style="text-align:center"><a href="images/face/95_scaled.jpg">Zum Bild</a></td>
        <td style="text-align:center"><a href="images/landscape/95_scaled.jpg">Zum Bild</a></td>
    </tr>
    <tr>
        <td style="text-align:center">Alle Skalierungsstufen (GIF)</td>
        <td style="text-align:center"><a href="images/cartoon/scaling_animated.gif">Zum Bild</a></td>
        <td style="text-align:center"><a href="images/face/scaling_animated.gif">Zum Bild</a></td>
        <td style="text-align:center"><a href="images/landscape/scaling_animated.gif">Zum Bild</a></td>
    </tr>
    <tr>
        <td style="text-align:center">Alle Komprimierungsstufen (GIF)</td>
        <td style="text-align:center"><a href="images/cartoon/compression_animated.gif">Zum Bild</a></td>
        <td style="text-align:center"><a href="images/face/compression_animated.gif">Zum Bild</a></td>
        <td style="text-align:center"><a href="images/landscape/compression_animated.gif">Zum Bild</a></td>
    </tr>
    <tr>
        <td style="text-align:center">Horizontal Cut (verschiedene Skalierungsstufen)</td>
        <td style="text-align:center"><a href="images/cartoon/horizontal_cut_scaling.png">Zum Bild</a></td>
        <td style="text-align:center"><a href="images/face/horizontal_cut_scaling.png">Zum Bild</a></td>
        <td style="text-align:center"><a href="images/landscape/horizontal_cut_scaling.png">Zum Bild</a></td>
    </tr>
    <tr>
        <td style="text-align:center">Horizontal Cut (verschiedene Komprimierungsstufen)</td>
        <td style="text-align:center"><a href="images/cartoon/horizontal_cut_compression.png">Zum Bild</a></td>
        <td style="text-align:center"><a href="images/face/horizontal_cut_compression.png">Zum Bild</a></td>
        <td style="text-align:center"><a href="images/landscape/horizontal_cut_compression.png">Zum Bild</a></td>
    </tr>
</table>

### Referenzen

Charrier, C., Maloney, L. T., Cherifi, H., & Knoblauch, K. (2007). Maximum likelihood difference scaling of image quality in compression-degraded images. _JOSA A, 24_(11), 3418-3426.

Ledig, C., Theis, L., Huszár, F., Caballero, J., Cunningham, A., Acosta, A., ... & Shi, W. (2017). Photo-realistic single image super-resolution using a generative adversarial network. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ (pp. 4681-4690).
