import argparse
import sys
import os
from typing import Optional, List, Iterator


# import av
import numpy as np
import torch
from torch import autocast
# import tensorflow as tf
from diffusers import PNDMScheduler, LMSDiscreteScheduler
from PIL import Image
from scipy.stats import chi
# from cog import BasePredictor, Input, Path

from animate import StableDiffusionAnimationPipeline
from matplotlib import pyplot 

# sys.path.append("/frame-interpolation")
# from eval import interpolator as film_interpolator, util as film_util

# MODEL_CACHE = "diffusers-cache"



# def setup():
#     """Load the model into memory to make running multiple predictions efficient"""
#     print("Loading pipeline...")
#     self.pipe = StableDiffusionAnimationPipeline.from_pretrained(
#         "CompVis/stable-diffusion-v1-4",
#         scheduler=make_scheduler(100),  # timesteps is arbitrary at this point
#         revision="fp16",
#         torch_dtype=torch.float16,
#         cache_dir=MODEL_CACHE,
#         local_files_only=True,
#     ).to("cuda")

#     # Stop tensorflow eagerly taking all GPU memory
#     gpus = tf.config.experimental.list_physical_devices("GPU")
#     for gpu in gpus:
#         tf.config.experimental.set_memory_growth(gpu, True)

#     print("Loading interpolator...")
#     self.interpolator = film_interpolator.Interpolator(
#         # from https://drive.google.com/drive/folders/1i9Go1YI2qiFWeT5QtywNFmYAA74bhXWj?usp=sharing
#         "/src/frame_interpolation_saved_model",
#         None,
#     )

# @torch.inference_mode()
# @torch.cuda.amp.autocast()
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt_start",
        type=str,
        default="",
        required=True,
        help="the start text prompt",
    )
    parser.add_argument(
        "--prompt_end",
        type=str,
        default="",
        required=True,
        help="the end text prompt",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=512,
        help="the image width"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=512,
        help="the image height"
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="the num inference steps"
    )
    parser.add_argument(
        "--prompt_strength",
        type=float,
        default=0.8,
        help="the strength of text",
    )
    parser.add_argument(
        "--num_animation_frames",
        type=int,
        default=10,
        help="the num of frames",
    )
    parser.add_argument(
        "--num_interpolation_steps",
        type=int,
        default=5,
        help="the num interpolation"
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="guidance_scale in diffusion"
    )
    parser.add_argument(
        "--gif_frames_per_second",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--output_format",
        type=str,
        default="gif",
        help="gif of video",
    )
    parser.add_argument(
        "--gif_ping_pong",
        default=False,
    )
    parser.add_argument(
        "--film_interpolation",
        default=False,
        help="the way of interpolation",
    )
    parser.add_argument(
        "-intermediate_output",
        default=True,
        help="save interpolation images"
    )
    args = parser.parse_args()
    return args


# def save_mp4(self, images, fps, width, height):
#     print("Saving MP4")
#     output_path = "/tmp/output.mp4"

#     output = av.open(output_path, "w")
#     stream = output.add_stream(
#         "h264", rate=fps, options={"crf": "17", "tune": "film"}
#     )
#     # stream.bit_rate = 8000000
#     # stream.bit_rate = 16000000
#     stream.width = width
#     stream.height = height

#     for i, image in enumerate(images):
#         image = (image * 255).astype(np.uint8)
#         image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#         frame = av.VideoFrame.from_ndarray(image, format="bgr24")
#         packet = stream.encode(frame)
#         output.mux(packet)

#     # flush
#     packet = stream.encode(None)
#     output.mux(packet)
#     output.close()

#     return Path(output_path)

def save_gif(pipe,images, fps):
    print("Saving GIF")
    pil_images = [
        pipe.numpy_to_pil(img.astype("float32"))[0] for img in images
    ]

    output_path = "/home/tli/mayuan/animate/spring_summer/video.gif"
    gif_frame_duration = int(1000 / fps)

    with open(output_path, "wb") as f:
        pil_images[0].save(
            fp=f,
            format="GIF",
            append_images=pil_images[1:],
            save_all=True,
            duration=gif_frame_duration,
            loop=0,
        )

    return 

def interpolate_latents(pipe,frames_latents, num_interpolation_steps):
    print("Interpolating images from latents")
    images = []
    for i in range(len(frames_latents) - 1):
        latents_start = frames_latents[i]
        latents_end = frames_latents[i + 1]
        for j in range(num_interpolation_steps):
            x = j / num_interpolation_steps
            latents = latents_start * (1 - x) + latents_end * x
            image = pipe.latents_to_image(latents)
            images.append(image)
    return images

# def interpolate_film(self, frames_latents, num_interpolation_steps):
#     print("Interpolating images with FILM")
#     images = [
#         self.pipe.latents_to_image(lat)[0].astype("float32")
#         for lat in frames_latents
#     ]
#     if num_interpolation_steps == 0:
#         return images

#     num_recursion_steps = max(int(np.ceil(np.log2(num_interpolation_steps))), 1)
#     images = film_util.interpolate_recursively_from_memory(
#         images, num_recursion_steps, self.interpolator
#     )
#     images = [img.clip(0, 1) for img in images]
#     return images


def make_scheduler(num_inference_steps, from_scheduler=None):
    scheduler = PNDMScheduler(
        beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear"
    )
    scheduler.set_timesteps(num_inference_steps, offset=1)
    if from_scheduler:
        scheduler.cur_model_output = from_scheduler.cur_model_output
        scheduler.counter = from_scheduler.counter
        scheduler.cur_sample = from_scheduler.cur_sample
        scheduler.ets = from_scheduler.ets[:]
        
    return scheduler


def slerp(t, v0, v1, DOT_THRESHOLD=0.9995):
    """helper function to spherically interpolate two arrays v1 v2"""
    # from https://gist.github.com/nateraw/c989468b74c616ebbc6474aa8cdd9e53

    if not isinstance(v0, np.ndarray):
        inputs_are_torch = True
        input_device = v0.device
        v0 = v0.cpu().numpy()
        v1 = v1.cpu().numpy()

    dot = np.sum(v0 * v1 / (np.linalg.norm(v0) * np.linalg.norm(v1)))
    if np.abs(dot) > DOT_THRESHOLD:
        v2 = (1 - t) * v0 + t * v1
    else:
        theta_0 = np.arccos(dot)
        sin_theta_0 = np.sin(theta_0)
        theta_t = theta_0 * t
        sin_theta_t = np.sin(theta_t)
        s0 = np.sin(theta_0 - theta_t) / sin_theta_0
        s1 = sin_theta_t / sin_theta_0
        v2 = s0 * v0 + s1 * v1

    if inputs_are_torch:
        v2 = torch.from_numpy(v2).to(input_device)

    return v2

def lerp(t,v0,v1):
    v2 = (1-t)*v0 + t*v1
    return v2

# def greate_circle(t,v0,v1,DOT_THRESHOLD=0.9995):
#     if not isinstance(v0, np.ndarray):
#         inputs_are_torch = True
#         input_device = v0.device
#         v0 = v0.cpu().numpy()
#         v1 = v1.cpu().numpy()
    
    
#     dot = np.sum(v0 * v1 / (np.linalg.norm(v0) * np.linalg.norm(v1)))
#     if np.abs(dot) > DOT_THRESHOLD:
#         v2 = (1 - t) * v0 + t * v1
#     else:
#         v0 /= np.linalg.norm(v0)
#         v1 /= np.linalg.norm(v1)
#         v1 = v0 - dot * v0
#         v1 /= np.linalg.norm(v1)
#         r = chi.rvs(df=118272/2)
#         z = np.cos(t * 2 * np.pi) * v0 + np.sin(t * 2 * np.pi) * v1
#         z *= r
#     if inputs_are_torch:
#         z = torch.from_numpy(z).to(input_device)
#     return z

def save_pil_image(image, path):
    image.save(path)
    return 

    
def main():
    args = parse_args()
    prompt_start = args.prompt_start
    prompt_end = args.prompt_end 
    width= args.width
    height = args.height
    num_inference_steps = args.num_inference_steps
    prompt_strength = args.prompt_strength
    num_animation_frames = args.num_animation_frames
    num_interpolation_steps = args.num_interpolation_steps
    guidance_scale = args.guidance_scale
    gif_frames_per_second = args.gif_frames_per_second
    gif_ping_pong = args.gif_ping_pong
    film_interpolation = args.film_interpolation
    intermediate_output = args.intermediate_output
    seed = args.seed
    output_format = args.output_format
    print("Loading pipeline...")
    YOUR_TOKEN="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"
    pipe = StableDiffusionAnimationPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    scheduler=make_scheduler(100),  # timesteps is arbitrary at this point
    revision="fp16",
    torch_dtype=torch.float16,
    use_auth_token=True,
    ).to("cuda")
    with torch.autocast("cuda"), torch.inference_mode():
        seed = args.seed
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")
        generator = torch.Generator("cuda").manual_seed(seed)
        batch_size = 1
        # Generate initial latents to start to generate animation frames from
        initial_scheduler = pipe.scheduler = make_scheduler(
            num_inference_steps
        )
        num_initial_steps = int(num_inference_steps * (1 - prompt_strength))
        print(f"Generating initial latents for {num_initial_steps} steps")
        initial_latents = torch.randn(
            (batch_size, pipe.unet.in_channels, height // 8, width // 8),
            generator=generator,
            device="cuda",
        )
        do_classifier_free_guidance = guidance_scale > 1.0
        text_embeddings_start = pipe.embed_text(
            prompt_start, do_classifier_free_guidance, batch_size
        )
        text_embeddings_end = pipe.embed_text(
            prompt_end, do_classifier_free_guidance, batch_size
        )
        text_embeddings_mid = slerp(0.5, text_embeddings_start, text_embeddings_end)
        latents_mid = pipe.denoise(
            latents=initial_latents,
            text_embeddings=text_embeddings_mid,
            t_start=1,
            t_end=num_initial_steps,
            guidance_scale=guidance_scale,
        )

        print("Generating first frame")
        # re-initialize scheduler
        pipe.scheduler = make_scheduler(num_inference_steps, initial_scheduler)
        latents_start = pipe.denoise(
            latents=latents_mid,
            text_embeddings=text_embeddings_start,
            t_start=num_initial_steps,
            t_end=None,
            guidance_scale=guidance_scale,
        )
        image_start = pipe.latents_to_image(latents_start)
        pipe.safety_check(image_start)

        if intermediate_output:
            save_pil_image(
                pipe.numpy_to_pil(image_start)[0], path="/home/tli/mayuan/animate/spring_summer/output-0.png"
            )

        print("Generating last frame")
        # re-initialize scheduler
        pipe.scheduler = make_scheduler(num_inference_steps, initial_scheduler)
        latents_end = pipe.denoise(
            latents=latents_mid,
            text_embeddings=text_embeddings_end,
            t_start=num_initial_steps,
            t_end=None,
            guidance_scale=guidance_scale,
        )
        image_end = pipe.latents_to_image(latents_end)
        pipe.safety_check(image_end)

        # Generate latents for animation frames
        frames_latents = []
        for i in range(num_animation_frames):
            if i == 0:
                latents = latents_start
            elif i == num_animation_frames - 1:
                latents = latents_end
            else:
                print(f"Generating frame {i}")
                text_embeddings = slerp(
                    i / (num_animation_frames - 1),
                    text_embeddings_start,
                    text_embeddings_end,
                )

                # re-initialize scheduler
                pipe.scheduler = make_scheduler(
                    num_inference_steps, initial_scheduler
                )
                latents = pipe.denoise(
                    latents=latents_mid,
                    text_embeddings=text_embeddings,
                    t_start=num_initial_steps,
                    t_end=None,
                    guidance_scale=guidance_scale,
                )

            # de-noise this frame
            frames_latents.append(latents)
            if intermediate_output and i > 0:
                image = pipe.latents_to_image(latents)
                save_pil_image(
                    pipe.numpy_to_pil(image)[0], path=f"/home/tli/mayuan/animate/spring_summer/output-{i}.png"
                )

        # Decode images by interpolate between animation frames
        if film_interpolation:
            # images = interpolate_film(frames_latents, num_interpolation_steps)
            print("no")
        else:
            images = interpolate_latents(
                pipe,frames_latents, num_interpolation_steps
            )

        # Save the video
        if gif_ping_pong:
            images += images[-1:1:-1]

        if output_format == "gif":
            save_gif(pipe,images, gif_frames_per_second)
        # else:
        #     save_mp4(images, gif_frames_per_second, width, height)


if __name__ == "__main__":
    main()

# """Run a single prediction on the model"""

# "tall rectangular black monolith, monkeys in the desert looking at a large tall monolith, a detailed matte painting by Wes Anderson, behance, light and space, reimagined by industrial light and magic, matte painting, criterion collection"
# "tall rectangular black monolith, a white room in the future with a bed, victorian details and a tall black monolith, a detailed matte painting by Wes Anderson, behance, light and space, reimagined by industrial light and magic, matte painting, criterion collection"
