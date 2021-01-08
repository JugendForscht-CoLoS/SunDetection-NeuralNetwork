"""sun_dataset dataset."""

import tensorflow_datasets as tfds
import os

# TODO(sun_dataset): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
Description is **formatted** as markdown.

It should also contain any processing which has been applied (if any),
(e.g. corrupted example skipped, images cropped,...):
"""

# TODO(sun_dataset): BibTeX citation
_CITATION = """
"""


class SunDataset(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for sun_dataset dataset."""
  MANUAL_DOWNLOAD_INSTRUCTIONS = "None"

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # TODO(sun_dataset): Specifies the tfds.core.DatasetInfo object
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            'image': tfds.features.Image(encoding_format='jpeg'),
            'segmentation_mask': tfds.features.Image(encoding_format='jpeg'),
        }),
        # If there's a common (input, target) tuple from the
        # features, specify them here. They'll be used if
        # `as_supervised=True` in `builder.as_dataset`.
        supervised_keys=None,  # e.g. ('image', 'label')
        homepage='https://dataset-homepage/',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager):
    data = dl_manager.manual_dir / 'suns'
    train = data / 'train'
    test = data / 'test'

    return {
        'train': self._generate_examples(img_path=train / 'images', mask_path=train / 'masks'),
        'test': self._generate_examples(img_path=test / 'images', mask_path=test / 'masks'),
    }

  def _generate_examples(self, img_path, mask_path):
      i = 0
      for img in os.listdir(img_path):
          name = "Image " + str(i)
          i += 1
          yield name, {
              'image': img_path / img,
              'segmentation_mask': mask_path / img,
          }
