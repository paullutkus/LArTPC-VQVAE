"""larcv dataset."""

import tensorflow_datasets as tfds
from os import walk


# TODO(larcv): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
Description is **formatted** as markdown.

It should also contain any processing which has been applied (if any),
(e.g. corrupted example skipped, images cropped,...):
"""

# TODO(larcv): BibTeX citation
_CITATION = """
"""


class Larcv(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for larcv dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # TODO(larcv): Specifies the tfds.core.DatasetInfo object
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            # These are the features of your dataset like images, labels ...
            'image': tfds.features.Image(shape=(64, 64, 1)),
            'label': tfds.features.ClassLabel(num_classes=1) 
        }),
        # If there's a common (input, target) tuple from the
        # features, specify them here. They'll be used if
        # `as_supervised=True` in `builder.as_dataset`.
        supervised_keys=('image', 'label'),  # e.g. ('image', 'label')
        homepage='http://deeplearnphysics.org/DataChallenge/',
        citation=None,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    # TODO(larcv): Downloads the data and defines the splits
    path = '/tf/data/cv/single_particle/' 

    # TODO(larcv): Returns the Dict[split names, Iterator[Key, Example]]
    return {
        'train': self._generate_examples(path + 'train/larcv_png_64/larcv_png_64/'),
        'test': self._generate_examples(path + 'test/larcv_png_64/larcv_png_64/')
    }

  def _generate_examples(self, path):
    """Yields examples."""
    # TODO(larcv): Yields (key, example) tuples from the dataset
    (_, _, filenames) = next(walk(path))
    for i, _ in enumerate(filenames):
        yield i, {'image': path + 'larcv_64_' + str(i) + '.png', 'label': 0}
