import numpy as np
import torch
import json
from autoencoder.AE import AE

from train import get_model, get_opts
from Batch import create_mask
from PIL import Image, ImageDraw

reference_data = np.array(json.loads(open('./dataset/train/run1.json').read()))
test_data = np.array(json.loads(open('./dataset/train/run2.json').read()))

image = Image.open('./web/public/assets/asphalt/Asphalt_001_COLOR.jpg')
draw = ImageDraw.Draw(image)
x_offset = 500
z_offset = 100
x_factor = 50
z_factor = 50


def draw_ellipse(x, z, r, color):
    # print('#### Drawing')
    # print(f'x: {x_offset + x * x_factor}, y: {z_offset + z * z_factor}')
    draw.ellipse((
        x_offset + x * x_factor,
        z_offset + z * z_factor,
        x_offset + x * x_factor+5,
        z_offset + z * z_factor+5), fill=color, outline='white')


def run_predict():
    opt = get_opts()
    # model = get_model(opt)
    model = AE()
    model.load_state_dict(torch.load(f'./saved_model/model.pt'))
    model.eval()

    def get_target(input, t_s, reference_data, jump):
        current_history = np.array(input)
        history_length = len(current_history)
        ref_loss = None
        ref_idx = 0
        for i in range(len(reference_data)-history_length-t_s):
            loss = np.abs(np.subtract(current_history,
                                      reference_data[i:i+history_length])).sum()

            if ref_loss is None or loss < ref_loss:
                ref_loss = loss
                ref_idx = i

        return reference_data[ref_idx+history_length+jump-1]

    def get_input_data(data, target_pos=None):
        current_position = data[0]
        source = [np.subtract(item, current_position)
                  for item in data[1:]]

        if target_pos is not None:
            source.append(np.subtract(target_pos, current_position))

        return source

    start = 30
    starting_points = reference_data[start:opt.w_s+start+1]
    for x, y, z, r in starting_points:
        draw_ellipse(x, z, r, color='red')

    for _ in range(0, 15):
        target_position = get_target(
            starting_points, opt.t_s, reference_data, opt.jump)
        x, y, z, r = target_position
        draw_ellipse(x, z, r, color='green')

        src = get_input_data(starting_points, target_position)
        src = torch.Tensor(np.array([src]))
        # src_mask = create_mask(src)

        ##########################################################
        b_s = src.size(dim=0)

        src_reshape = torch.reshape(src, (b_s, (opt.w_s + 1)  * 4))

        preds = model(src_reshape)
        preds = torch.reshape(preds, (b_s, opt.t_s, 4))
        preds_array = preds.tolist()

        for i in range(0, len(preds_array[0])):
            preds_array[0][i] = np.add(preds_array[0][i], starting_points[0]).tolist()
            x, y, z, r = preds_array[0][i]
            draw_ellipse(x, z, r, color='blue')

        starting_points = preds_array[0]
        ##########################################################
        
        # e_outputs = model.encoder(src, src_mask)

        # outputs = torch.empty(1, opt.w_s + 1, src.size(dim=2))
        # outputs[0][0] = torch.Tensor(np.zeros(4, dtype=float))

        # for i in range(1, opt.w_s + 1):
        #     trg_mask = np.triu(np.ones((1, i, i)), k=1).astype('uint8')
        #     trg_mask = torch.from_numpy(trg_mask) == 0

        #     out = model.out(model.decoder(torch.unsqueeze(
        #         outputs[0][:i], dim=0), e_outputs, src_mask, trg_mask))

        #     outputs[0][i] = out[0][i-1]

        #     absolute_pos = np.add(out[0][i-1].tolist(), starting_points[0])
        #     x, y, z, r = absolute_pos
        #     draw_ellipse(x, z, r, color='blue')

        # outputs[0][0] = torch.Tensor(starting_points[-1])
        # for i in range(1, len(outputs[0])):
        #     outputs[0][i] = torch.Tensor(
        #         np.add(outputs[0][i].tolist(), starting_points[0]))

        # starting_points = outputs[0].tolist()


run_predict()
image.save('./test.png')
