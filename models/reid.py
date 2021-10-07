from torchreid.utils import FeatureExtractor
class ReidFeatExtractor:
    def __init__(self, model_path="/home/ixti/Documents/projects/github/my_tracker/checkpoints/osnet_d_m_c.pth.tar", model_type="osnet_x1_0") -> None:
        self.model_path = model_path
        self.model_type= model_type
        self.extractor = FeatureExtractor(
            model_name=self.model_type, 
            model_path=self.model_path,
            device='cuda'
        )
    def __call__(self,imgs):
        return self.run(imgs)
    def run(self, imgs):
        features = self.extractor(imgs)
        return features