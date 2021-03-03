# Sonnenerkennung
Unsere Sonnenerkennung baut auf einem neuronalen Netz auf.

## Datensatz
Für das neuronales Netz haben wir Sonnenbilder gesammelt und dazugehörige Masken gemalt.
Um die Daten (Sonnenbilder und Masken) benutzen zu können, haben wir in dem Ordner **sun_dataset** in der Datei **sun_dataset.py** ein Tensorflow Datensatz erstellt. Dabei werden die Daten eingelesen und in train und test unterteilt.
Um eine größere Bandbreite an Daten abzudecken benutzen wir Data Augmentation. Dabei werden die ursprünglichen Daten leicht verändert. In **augment.py** könnnen Sie sich das genauer ansehen.

## Training
