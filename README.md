# Sonnenerkennung
Unsere Sonnenerkennung baut auf einem neuronalen Netz auf.

## Datensatz
Für das neuronales Netz haben wir Sonnenbilder gesammelt und dazugehörige Masken gemalt.
Um die Daten (Sonnenbilder und Masken) benutzen zu können, haben wir in dem Ordner **sun_dataset** in der Datei **sun_dataset.py** ein Tensorflow Datensatz erstellt. Dabei werden die Daten eingelesen und in train und test unterteilt.
Um eine größere Bandbreite an Daten abzudecken benutzen wir Data Augmentation. Dabei werden die ursprünglichen Daten leicht verändert. In **augment.py** könnnen Sie sich das genauer ansehen.

## Training
Das Training des neuronalen Netzes erfolgt vorwiegend in **main.py**. Hier wird das Dataset zuerst prepariert. Anschließend wird das Model aus **model/model.py** geladen. Wir benutzen eine selbstgeschriebene UNET-Struktur. Es ist aber noch in Planung, dass wir andere Strukturen testen. 
Dann wird das Model kompilliert. Vor allem an der Loss-Function (Fehlerfunktion) haben wir uns lange versucht. Schlussendlich haben wir uns für den Tversky-Loss entschieden (siehe **losses/losses.py**). Die letzten Schritte des neuronalen Netzes sind das Training und die Speicherung.

##Visualisierung
