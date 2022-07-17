# Handpainted-CG-Diffusion
FeiArt_Handpainted CG Diffusion

FeiArt Handpainted CG Diffusion is a custom diffusion model trained by @FeiArt_AiArt.


This is a huge 512*512 custom diffusion model, trained with A100.
It can be used to create Handpainted CG style images. and it can also create stylized portraits.

And it runs within a fork of DiscoDiffusion 5.6.So you can use all DD's basic functions.

If you create a fun image with this model, please show your result and message us on twitter at @FeiArt_AiArt (https://twitter.com/FeiArt_AiArt)

Or you can join the FeiArt Diffusion Discord (https://discord.gg/5DzGHdSGcH)

Share your work created with this model. Exchange experiences and parameters. And see more interesting custom models

Copying the model file directly will not work if someone tries to run the notebook locally.

there is one thing to note. I used classcond for this model at the time of training, but it actually ended up being used as an unconditional model. 
So in addition to copying all the model settings, you also need to add this code to the run module.

print('Prepping model...')
model, diffusion = create_model_and_diffusion(**model_config)
if diffusion_model == 'FeiArt_Handpainted_CG_Diffusion':
    model.load_state_dict(torch.load(f'{model_path}/{get_model_filename(diffusion_model)}', map_location='cpu'),strict=False)
