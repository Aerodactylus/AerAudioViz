from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
import pandas as pd
from tqdm import tqdm
from typing import Optional

from aeraudioviz.audio import Audio
from aeraudioviz.audio.feature_utils import normalise_features
from aeraudioviz.image import ImageModifiers, BaseImage
from aeraudioviz.video.modifier_mapping import ModifierMapping


class VideoGenerator:

    # ToDo: consider how to ensure the feature_time_series index freq and fps are consistent and only require
    #  specification at one point

    def __init__(
            self,
            base_image: BaseImage,
            feature_time_series: pd.DataFrame,
            modifier_mappings: tuple[ModifierMapping],
            fps: int = 24,
            audio: Optional[Audio] = None,
            normalise_feature_values: bool = True
    ):
        self.base_image = base_image
        self.feature_time_series = feature_time_series
        if normalise_feature_values:
            self.feature_time_series = normalise_features(self.feature_time_series)
        self.modifier_mappings = modifier_mappings
        self.image_modifier = ImageModifiers()
        self.fps = fps
        self.audio = audio

    def generate(self, output_path: str = "output_video.mp4"):
        frames = []
        print("Generating video frames...")
        for idx, row in tqdm(self.feature_time_series.iterrows(), total=len(self.feature_time_series)):
            frame = self.base_image.rgb_image.copy()
            for mod_mapping in self.modifier_mappings:
                mod_value = row[mod_mapping.modifier_column]
                kwargs = {
                    arg: mod_mapping.kwarg_ranges[arg][0] + (
                            mod_mapping.kwarg_ranges[arg][1] - mod_mapping.kwarg_ranges[arg][0]
                    ) * mod_value
                    for arg in mod_mapping.kwarg_ranges.keys()
                }
                frame = mod_mapping.modifier_function(frame, **kwargs)
            frames.append(frame)
        clip = ImageSequenceClip(frames, fps=24)
        print("Frames generated.")
        if self.audio is not None:
            print("Setting audio...")
            clip = clip.set_audio(self.audio.moveipy_audio_clip)
            print("Audio set.")
        print("Writing video...")
        clip.write_videofile(output_path, codec="libx264")
        print(f"Video written successfully to {output_path}.")
