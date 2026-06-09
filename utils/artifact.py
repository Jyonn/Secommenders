from pathlib import Path


class ArtifactStore:
    ROOT = Path('artifacts')

    def __init__(self, dataset: str):
        self.dataset = dataset.lower()

    def _dir(self, *parts: str) -> Path:
        path = self.ROOT.joinpath(*parts)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def formatted_dir(self) -> Path:
        return self._dir('formatted', self.dataset)

    def processed_dir(self) -> Path:
        return self._dir('processed', self.dataset)

    def embedded_dir(self, model: str) -> Path:
        return self._dir('embedded', self.dataset, model)

    def quantized_dir(self, model: str, quantizer_name: str | None = None) -> Path:
        if quantizer_name:
            return self._dir('quantized', self.dataset, model, quantizer_name)
        return self._dir('quantized', self.dataset, model)

    def clustered_dir(self, prepare_id: str) -> Path:
        return self._dir('clustered', self.dataset, prepare_id)

    def compiled_dir(self, prepare_id: str) -> Path:
        return self._dir('compiled', self.dataset, prepare_id)

    def prepared_dir(self, prepare_id: str) -> Path:
        return self.compiled_dir(prepare_id)

    def trained_dir(self, run_id: str) -> Path:
        return self._dir('trained', self.dataset, run_id)
