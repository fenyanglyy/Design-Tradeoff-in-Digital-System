import time
import math

from bitarray import bitarray
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.fftpack import dct, idct

from Block import Block, pad_frame, PSNR, SSIM, convert_to_uint8
import Decoder
import EntropyCoding
from GlobalConfig import Config
from YUV_formater import YUV_Operator

config = Config()

class Encoder:
    def __init__(self) -> None:
        self.config = config
        self.yuv_operator = YUV_Operator(self.config)
        self.entropy_coder = EntropyCoding.EntropyCoder()
    

    def _construct_frame(self, frame_shape, value):
        virtual_frame = np.full(frame_shape, value, dtype=np.int16)
        return virtual_frame
    
    def _multiple_inter_predict_Y_(self, block:Block, reference_frame_list, search_range, current_mvp):
        if len(reference_frame_list) == 0:
            return
        best_mae = float('inf')
        motion_vector = np.array([0, 0], dtype=np.int8)
        best_matching_block = block
        best_matching_index = -1
        for i, reference_frame in enumerate(reference_frame_list):
            temp_motion_vector, match_block, temp_mae = self._simple_inter_predict_Y_(block, reference_frame, search_range, current_mvp)
            if temp_mae < best_mae and temp_mae - best_mae < -0.5:
                best_mae = temp_mae
                motion_vector = temp_motion_vector
                best_matching_block = match_block
                best_matching_index = i  

            if temp_mae < best_mae and temp_mae - best_mae >= -0.5:
                if abs(temp_motion_vector[0]) < abs(motion_vector[0]):
                    motion_vector = temp_motion_vector
                    best_matching_block = match_block
                    best_matching_index = i
                elif abs(temp_motion_vector[0]) == abs(motion_vector[0]):
                    if abs(temp_motion_vector[1]) < abs(motion_vector[1]):
                        motion_vector = temp_motion_vector
                        best_matching_block = match_block
                        best_matching_index = i
        return best_matching_index, motion_vector, best_matching_block, best_mae


    def _add_original_candidate_(self, centre_block:Block, reference_frame, search_range):
        candidate_blocks = [[None for i in range((2*search_range+1))] for j in range((2*search_range+1))]
        frame_height, frame_width = reference_frame.shape
        for dx in range(0, 2*search_range+1):
            for dy in range(0, 2*search_range+1):
                temp_x = centre_block.x + dx - search_range
                temp_y = centre_block.y + dy - search_range
                if temp_x >= 0 and temp_x <= frame_height - centre_block.block_size \
                    and temp_y >= 0 and temp_y <= frame_width- centre_block.block_size:
                    temp_block = Block(temp_x, temp_y, centre_block.block_size, reference_frame)
                    candidate_blocks[dx][dy] = temp_block.block_data.copy()
        return candidate_blocks

    
    def _add_fractional_candidate_(self, candidate_blocks):

        # scale candidate blocks 
        scaled_candidate_blocks = [[None for i in range(2*len(candidate_blocks)-1)] for j in range(2*len(candidate_blocks[0])-1)]
        for i in range(len(candidate_blocks)):
            for j in range(len(candidate_blocks[i])):
                if candidate_blocks[i][j] is not None:
                    new_block_data = candidate_blocks[i][j]
                    scaled_candidate_blocks[2*i][2*j] = new_block_data.copy()

        # add fractional rows
        for i in range(0, len(scaled_candidate_blocks), 2):
            for j in range(1, len(scaled_candidate_blocks[i]), 2):
                if scaled_candidate_blocks[i][j-1] is not None and scaled_candidate_blocks[i][j+1] is not None:
                    new_block_data = (scaled_candidate_blocks[i][j-1]+scaled_candidate_blocks[i][j+1])//2
                    scaled_candidate_blocks[i][j] = new_block_data.copy()

        # add fractional cols
        for i in range(1, len(scaled_candidate_blocks), 2):
            for j in range(0, len(scaled_candidate_blocks[i]), 2):
                if scaled_candidate_blocks[i-1][j] is not None and scaled_candidate_blocks[i+1][j] is not None:
                    new_block_data = (scaled_candidate_blocks[i-1][j]+scaled_candidate_blocks[i+1][j])//2
                    scaled_candidate_blocks[i][j] = new_block_data.copy()

        # add fractional inner
        for i in range(1, len(scaled_candidate_blocks), 2):
            for j in range(1, len(scaled_candidate_blocks[i]), 2):
                if scaled_candidate_blocks[i-1][j-1] is not None \
                and scaled_candidate_blocks[i+1][j-1] is not None \
                and scaled_candidate_blocks[i-1][j+1] is not None \
                and scaled_candidate_blocks[i+1][j+1] is not None:
                    new_block_data = \
                        (scaled_candidate_blocks[i-1][j-1]
                         +scaled_candidate_blocks[i+1][j-1]
                         +scaled_candidate_blocks[i-1][j+1]
                         +scaled_candidate_blocks[i+1][j+1])//4
                    scaled_candidate_blocks[i][j] = new_block_data

        return scaled_candidate_blocks

    def _simple_inter_predict_Y_(self, block:Block, reference_frame, search_range, current_mvp):
        # Feature enalbled
        FMEEnable = self.config.FMEEnable
        FastME = self.config.FastME

        # Initial value
        best_matching_block = Block(block.x, block.y, block.block_size, reference_frame)
        best_mae = np.abs(block.block_data.astype(np.int16) - best_matching_block.block_data.astype(np.int16)).mean()
        motion_vector = np.array([0, 0], dtype=np.int8)

        if FMEEnable == True:
            if current_mvp[0] % 2 != 0:
                dx = (current_mvp[0]+1)//2
            else:
                dx = current_mvp[0]//2
            if current_mvp[1] % 2 != 0:
                dy = (current_mvp[1]+1)//2
            else:
                dy = current_mvp[1]//2
        else:
            dx = current_mvp[0]
            dy = current_mvp[1]

        # add candidate block
        if FastME == True:
            real_search_range = 1
            centre_block = Block(block.x + dx, block.y + dy, block.block_size, reference_frame)
        else:
            real_search_range = search_range
            centre_block = Block(block.x, block.y, block.block_size, reference_frame)

        candidate_blocks = self._add_original_candidate_(centre_block, reference_frame, real_search_range)

        if FMEEnable == True:
            real_search_range = real_search_range*2
            candidate_blocks = self._add_fractional_candidate_(candidate_blocks)

        for i in range(len(candidate_blocks)):
            for j in range(len(candidate_blocks[i])):
                if FastME:
                    if FMEEnable == True:
                        if current_mvp[0] % 2 != 0:
                            dx = (current_mvp[0]+1)
                        else:
                            dx = current_mvp[0]
                        if current_mvp[1] % 2 != 0:
                            dy = (current_mvp[1]+1)
                        else:
                            dy = current_mvp[1]
                    else:
                        dx = current_mvp[0]
                        dy = current_mvp[1]
                else:
                    dx = 0
                    dy = 0
                # frame boundary
                if candidate_blocks[i][j] is not None:
                    temp_block_data = candidate_blocks[i][j]
                    temp_mae = np.abs(block.block_data.astype(np.int16) - temp_block_data.astype(np.int16)).mean()

                    if temp_mae < best_mae and temp_mae - best_mae < -0.5:
                        best_mae = temp_mae
                        motion_vector = np.array([i-real_search_range+dx, j-real_search_range+dy], dtype=np.int8)
                        best_matching_block = Block(block.x+motion_vector[0], block.y+motion_vector[1], block.block_size, None, temp_block_data)

                    if temp_mae < best_mae and temp_mae - best_mae > -0.5:
                        if abs(i-real_search_range) < abs(motion_vector[0]):
                            motion_vector = np.array([i-real_search_range+dx, j-real_search_range+dy], dtype=np.int8)
                            best_matching_block = Block(block.x+motion_vector[0], block.y+motion_vector[1], block.block_size, None, temp_block_data)
                        elif abs(i-real_search_range+centre_block.x-block.x) == abs(motion_vector[0]) and abs(j-real_search_range+centre_block.y-block.y) < abs(motion_vector[1]):
                            motion_vector = np.array([i-real_search_range+dx, j-real_search_range+dy], dtype=np.int8)
                            best_matching_block = Block(block.x+motion_vector[0], block.y+motion_vector[1], block.block_size, None, temp_block_data)
        # print(motion_vector)
        return motion_vector, best_matching_block, best_mae
    

    def _approximate_residual_block(self, residual_block:Block):
        factor = 2**self.config.residual_n
        residual_block.block_data = np.int16(np.round(residual_block.block_data/factor) * factor)
        return residual_block


    # Decide the frame mode by I_period
    # First Frame is always I_frame, then k-1 frames are P
    def _decide_frame_mode(self, k):
        if k % self.config.I_Period == 0:
            return 'I'
        else:
            return 'P'
        

    def _reconstruct_block_(self, predicted_block:Block, residual_block:Block):
        return Block(residual_block.x, residual_block.y, residual_block.block_size, \
                     None, (predicted_block.block_data+residual_block.block_data).clip(0, 255))


    def dct_transform(self, block):
        return np.round(dct(dct(block, axis=0, norm='ortho'), axis=1, norm='ortho')).astype(np.int16)
    

    def idct_transform(self, block):
        return np.round(idct(idct(block, axis=0, norm='ortho'), axis=1, norm='ortho')).astype(np.int16)
    

    def quantize(self, coefficients, QP, i):
        Q = self.calculate_quantization_matrix(QP, i)
        return np.round(coefficients / Q).astype(np.int16)
    

    def calculate_quantization_matrix(self, QP, i):
        Q = np.zeros((i, i), dtype=float)
        for x in range(i):
            for y in range(i):
                if (x + y) < (i - 1):
                    Q[x][y] = 2 ** QP
                elif (x + y) == (i - 1):
                    Q[x][y] = 2 ** (QP + 1)
                else:
                    Q[x][y] = 2 ** (QP + 2)
        return Q
    

    def transform_block(self, block_data, QP, block_size):
        # 1. 对每个块执行DCT变换
        dct_coefficients = self.dct_transform(block_data)
        # 2. 量化DCT系数
        quantized_coefficients = self.quantize(dct_coefficients, QP, block_size)
        return quantized_coefficients
    

    # 实现I帧的内部预测（水平或垂直）
    def _intra_predict_Y_(self, block, reference_frame):
        # 选择水平或垂直模式
        horizontal_mode = self._horizontal_intra_prediction_(block, reference_frame)
        vertical_mode = self._vertical_intra_prediction_(block, reference_frame)

        # 选择MAE成本最低的模式
        if horizontal_mode['MAE'] < vertical_mode['MAE']:
            selected_mode = horizontal_mode
        else:
            selected_mode = vertical_mode

        return selected_mode

    # 实现水平内部预测
    def _horizontal_intra_prediction_(self, block:Block, reference_frame):
        # 水平预测逻辑：使用左边的重建样本进行预测
        if block.y == 0:
            left_border = np.full((block.block_size, 1), 128, dtype=np.int16)  # 使用128填充缺失的样本
        else:
            left_border = reference_frame[block.x:block.x+block.block_size, block.y-1:block.y]
        predicted_block_data = np.tile(left_border, (1, block.block_size))
        predicted_block = Block(block.x, block.y, block.block_size, None, predicted_block_data)
        mae_cost = np.abs(block.block_data - predicted_block.block_data).mean()
        return {'mode': 0, 'MAE': mae_cost, 'predicted_block': predicted_block}


    # 实现垂直内部预测
    def _vertical_intra_prediction_(self, block:Block, reference_frame):
        # 垂直预测逻辑：使用顶部的重建样本进行预测
        if block.x == 0:
            top_border = np.full((1, block.block_size), 128, dtype=np.int16)  # 使用128填充缺失的样本
        else:
            top_border = reference_frame[block.x-1:block.x, block.y:block.y+block.block_size]
        predicted_block_data = np.tile(top_border, (block.block_size, 1))
        predicted_block = Block(block.x, block.y, block.block_size, None, predicted_block_data)
        mae_cost = np.abs(block.block_data - predicted_block.block_data).mean()
        return {'mode': 1, 'MAE': mae_cost, 'predicted_block': predicted_block}



    def predict(self, block:Block, frame_mode, reference_frame_buffer, reconstructed_frame, current_mvp):
        if frame_mode == 'P':
            i, motion_vector, predicted_block, mae = self._multiple_inter_predict_Y_(block, reference_frame_buffer, self.config.search_range, current_mvp)
            extra_data = np.array([motion_vector[0], motion_vector[1], i])
        elif frame_mode == 'I':
            selected_mode = self._intra_predict_Y_(block, reconstructed_frame)
            predicted_block = selected_mode['predicted_block']
            mae = selected_mode['MAE']
            mode = selected_mode['mode']
            extra_data = mode
        return predicted_block, mae, extra_data
    

    def VBS_predict(self, block:Block, frame_mode, reference_frame_buffer, reconstructed_frame, current_mvp, current_mv, current_I_mode):
        decoder = Decoder.Decoder(self.config)
        block_data_lengths = bitarray()
        split_block_data_lengths = bitarray()
        new_reconstructed_frame = reconstructed_frame.copy()
        split = False
        if frame_mode == 'P':
            # no split
            i, motion_vector, predicted_block, mae = self._multiple_inter_predict_Y_(block, reference_frame_buffer, self.config.search_range, current_mvp)
            residual_block = Block(block.x, block.y, block.block_size, None, block.block_data-predicted_block.block_data)
            coff = self.transform_block(residual_block.block_data, self.config.QP, block.block_size)
            extra_data = np.array([motion_vector[0], motion_vector[1], i])
            mv_differential = extra_data - current_mv
            encoded_block_data = []
            encoded_block_data.extend(mv_differential)
            encoded_block_data.extend(self.entropy_coder.reorder(coff, block.block_size))
            encoded_block_data = self.entropy_coder.rle(encoded_block_data)
            encoded_block_data = self.entropy_coder.golomb_encode_array(encoded_block_data, self.config.golomb_m)
            r_bits = len(encoded_block_data)
            block_data_lengths.extend(bitarray(format(len(encoded_block_data), 'b').zfill(16)))

            # reconstrcut
            residual_block_data = decoder.inverse_transform_block_data(coff, self.config.QP, block.block_size)
            inversed_residual_block = Block(block.x, block.y, block.block_size, None, residual_block_data)
            reconstructed_block = self._reconstruct_block_(predicted_block=predicted_block, residual_block=inversed_residual_block)
            distortion_before_split = PSNR(block.block_data, reconstructed_block.block_data)
            # distortion_before_split = np.sum(np.abs(block.block_data-predicted_block.block_data))
            vbs_lambda = self.config.VBS_lambda_constant*2**((self.config.QP-12)/3)
            rdo_before_split = -distortion_before_split + vbs_lambda*r_bits
            
            # split
            reconstrcuted_block_data = np.zeros((block.block_size, block.block_size), dtype=np.int16)
            split_total_encoded_data = bitarray()
            split_predicted_block_data = np.zeros((block.block_size, block.block_size), dtype=np.int16)
            split_block_size = block.block_size//2
            distortion_split = 0
            steps = [0, block.block_size//2]
            split_r_bits = 0
            split_mae_total = 0
            split_current_mv = current_mv
            for i in steps:
                for j in steps:
                    split_encoded_block_data = []
                    split_block = Block(block.x + i, block.y + j, split_block_size, None, block.block_data[i: i+split_block_size, j: j+split_block_size])
                    split_i, split_motion_vector, split_predicted_block, split_mae = self._multiple_inter_predict_Y_(split_block, reference_frame_buffer, self.config.search_range, current_mvp)
                    split_mae_total += split_mae
                    split_predicted_block_data[i:i+split_block_size, j:j+split_block_size] = split_predicted_block.block_data
                    split_residual = Block(block.x, block.y, split_block_size, None, split_block.block_data-split_predicted_block.block_data)
                    if self.config.QP == 0:
                        qp = 0
                    else:
                        qp = self.config.QP-1
                    split_coff = self.transform_block(split_residual.block_data, qp, split_block_size)
                    split_extra_data = np.array([split_motion_vector[0], split_motion_vector[1], split_i])
                    split_mv_differential = split_extra_data - split_current_mv
                    split_current_mv = split_extra_data
                    split_encoded_block_data.extend(split_mv_differential)
                    split_encoded_block_data.extend(self.entropy_coder.reorder(split_coff, split_block.block_size))
                    split_encoded_block_data = self.entropy_coder.rle(split_encoded_block_data)
                    split_encoded_block_data = self.entropy_coder.golomb_encode_array(split_encoded_block_data, self.config.golomb_m)
                    split_r_bits += len(split_encoded_block_data)
                    split_block_data_lengths.extend(bitarray(format(len(split_encoded_block_data), 'b').zfill(16)))
                    split_total_encoded_data.extend(split_encoded_block_data.copy())

                    # reconstrcut
                    split_residual_block_data = decoder.inverse_transform_block_data(split_coff, qp, split_block_size)
                    split_inversed_residual_block = Block(block.x, block.y, split_block_size, None, split_residual_block_data)
                    split_reconstructed_block = self._reconstruct_block_(predicted_block=split_predicted_block, residual_block=split_inversed_residual_block)
                    reconstrcuted_block_data[i:i+split_block_size, j:j+split_block_size] = split_reconstructed_block.block_data
                    # distortion_split += np.sum(np.abs(split_block.block_data-split_predicted_block.block_data))
                    distortion_split += PSNR(split_block.block_data, split_reconstructed_block.block_data)
            
            split_reconstructed_block = Block(block.x, block.y, block.block_size, None, reconstrcuted_block_data)
            vbs_lambda_split = self.config.VBS_lambda_constant*2**((self.config.QP-12)/3)
            rdo_split = -distortion_split/4 + vbs_lambda_split*split_r_bits

            # print(f'{distortion_before_split}=')
            # print(f'{distortion_split}=')
            # print(f'{r_bits}=')
            # print(f'{split_r_bits}=')
            
            # need spliting blocks
            if rdo_split < rdo_before_split:
                predicted_block = Block(block.x, block.y, block.block_size, None, split_predicted_block_data)
                encoded_block_data = split_total_encoded_data
                new_current_mv = split_current_mv
                mae = split_mae_total/4
                split = True
                reconstructed_block = split_reconstructed_block
                block_data_lengths = split_block_data_lengths
            else:
                new_current_mv = extra_data
            
            return split, predicted_block, reconstructed_block, encoded_block_data, new_current_mv, block_data_lengths, mae

        if frame_mode == 'I':
            selected_mode = self._intra_predict_Y_(block, new_reconstructed_frame)
            predicted_block = selected_mode['predicted_block']
            mae = selected_mode['MAE']
            mode = selected_mode['mode']
            extra_data = mode

            # no split
            residual_block = Block(block.x, block.y, block.block_size, None, block.block_data-predicted_block.block_data)
            coff = self.transform_block(residual_block.block_data, self.config.QP, block.block_size)
            mode_differential = extra_data - current_I_mode
            encoded_block_data = []
            encoded_block_data.append(mode_differential)
            encoded_block_data.extend(self.entropy_coder.reorder(coff, block.block_size))
            encoded_block_data = self.entropy_coder.rle(encoded_block_data)
            encoded_block_data = self.entropy_coder.golomb_encode_array(encoded_block_data, self.config.golomb_m)
            r_bits = len(encoded_block_data)
            block_data_lengths.extend(bitarray(format(len(encoded_block_data), 'b').zfill(16)))

            # reconstrcut
            residual_block_data = decoder.inverse_transform_block_data(coff, self.config.QP, block.block_size)
            inversed_residual_block = Block(block.x, block.y, block.block_size, None, residual_block_data)
            reconstructed_block = self._reconstruct_block_(predicted_block=predicted_block, residual_block=inversed_residual_block)
            distortion_before_split = PSNR(block.block_data, reconstructed_block.block_data)
            # distortion_before_split = np.sum(np.abs(block.block_data-predicted_block.block_data))
            vbs_lambda = self.config.VBS_lambda_constant*2**((self.config.QP-12)/3)
            rdo_before_split = -distortion_before_split + vbs_lambda*r_bits
            
            # split
            reconstrcuted_block_data = np.zeros((block.block_size, block.block_size), dtype=np.int16)
            split_total_encoded_data = bitarray()
            split_predicted_block_data = np.zeros((block.block_size, block.block_size), dtype=np.int16)
            split_block_size = block.block_size//2
            distortion_split = 0
            steps = [0, block.block_size//2]
            split_r_bits = 0
            split_mae_total = 0
            split_current_I_mode = current_I_mode
            for i in steps:
                for j in steps:
                    split_encoded_block_data = []
                    split_block = Block(block.x + i, block.y + j, split_block_size, None, block.block_data[i: i+split_block_size, j: j+split_block_size])
                    selected_mode = self._intra_predict_Y_(split_block, new_reconstructed_frame)
                    split_predicted_block = selected_mode['predicted_block']
                    split_mae = selected_mode['MAE']
                    split_mode = selected_mode['mode']
                    split_extra_data = split_mode

                    split_mae_total += split_mae
                    split_predicted_block_data[i:i+split_block_size, j:j+split_block_size] = split_predicted_block.block_data
                    split_residual = Block(block.x, block.y, split_block_size, None, split_block.block_data-split_predicted_block.block_data)
                    if self.config.QP == 0:
                        qp = 0
                    else:
                        qp = self.config.QP-1
                    split_coff = self.transform_block(split_residual.block_data, qp, split_block_size)
                    split_mode_differential = split_extra_data - split_current_I_mode
                    split_current_I_mode = split_extra_data
                    split_encoded_block_data.append(split_mode_differential)
                    split_encoded_block_data.extend(self.entropy_coder.reorder(split_coff, split_block.block_size))
                    split_encoded_block_data = self.entropy_coder.rle(split_encoded_block_data)
                    split_encoded_block_data = self.entropy_coder.golomb_encode_array(split_encoded_block_data, self.config.golomb_m)
                    split_r_bits += len(split_encoded_block_data)
                    split_block_data_lengths.extend(bitarray(format(len(split_encoded_block_data), 'b').zfill(16)))
                    split_total_encoded_data.extend(split_encoded_block_data.copy())

                    # reconstrcut
                    split_residual_block_data = decoder.inverse_transform_block_data(split_coff, qp, split_block_size)
                    split_inversed_residual_block = Block(block.x, block.y, split_block_size, None, split_residual_block_data)
                    split_reconstructed_block = self._reconstruct_block_(predicted_block=split_predicted_block, residual_block=split_inversed_residual_block)
                    reconstrcuted_block_data[i:i+split_block_size, j:j+split_block_size] = split_reconstructed_block.block_data
                    new_reconstructed_frame[block.x+i:block.x+i+split_block_size, block.y+j:block.y+j+split_block_size] = split_reconstructed_block.block_data
                    # distortion_split += np.sum(np.abs(split_block.block_data-split_predicted_block.block_data))
                    distortion_split += PSNR(split_block.block_data, split_reconstructed_block.block_data)
            
            split_reconstructed_block = Block(block.x, block.y, block.block_size, None, reconstrcuted_block_data)
            vbs_lambda_split = self.config.VBS_lambda_constant*2**((self.config.QP-12)/3)
            rdo_split = -distortion_split/4 + vbs_lambda_split*split_r_bits

            # print(f'{distortion_before_split}=')
            # print(f'{distortion_split}=')
            # print(f'{r_bits}=')
            # print(f'{split_r_bits}=')
            
            # need spliting blocks
            if rdo_split < rdo_before_split:
                predicted_block = Block(block.x, block.y, block.block_size, None, split_predicted_block_data)
                encoded_block_data = split_total_encoded_data
                new_current_I_mode = split_current_I_mode
                mae = split_mae_total/4
                split = True
                reconstructed_block = split_reconstructed_block
                block_data_lengths = split_block_data_lengths
            else:
                new_current_I_mode = extra_data
            
            return split, predicted_block, reconstructed_block, encoded_block_data, new_current_I_mode, block_data_lengths, mae

    # header按字节顺序依次为
    # 2B blocks_counts
    # 2B width
    # 2B height
    # 1B block_size
    # 1B QP
    # 1B golomb_m 
    # 5bits nrefFrames
    # 1bit vbs
    # 1bit fractionalme
    # 1bit fastme
    def dump_header(self, file):
        blocks_counts = 0
        for i in range(0, self.config.height, self.config.block_size): 
            for j in range(0, self.config.width, self.config.block_size):  
                blocks_counts += 1
        header = bitarray()
        header.extend(format(blocks_counts, 'b').zfill(16))
        header.extend(format(self.config.width, 'b').zfill(16))
        header.extend(format(self.config.height, 'b').zfill(16))
        header.extend(format(self.config.block_size, 'b').zfill(8))
        header.extend(format(self.config.QP, 'b').zfill(8))
        header.extend(format(self.config.golomb_m, 'b').zfill(8))
        header.extend(format(self.config.nRefFrames, 'b').zfill(5))
        header.extend(format(self.config.VBSEnable, 'b').zfill(1))
        header.extend(format(self.config.FMEEnable, 'b').zfill(1))
        header.extend(format(self.config.FastME, 'b').zfill(1))

        # dump prediction feature

        file.write(header)


    
    def encode(self, input_video_path, ws = 'Assignment/encoder_ws'):
        # Read from video, default 420p
        frames = self.yuv_operator.read_yuv(input_video_path)
        block_size = self.config.block_size
        width = self.config.width
        height = self.config.height
        decoder = Decoder.Decoder(config)
        total_bitrates = []
        total_time = []
        split_percentage = []
        psnr = []

        with open(f"{ws}/encoded_data.13", "wb+") as file:
            file.truncate()
            self.dump_header(file)
        
        
        # initialize shared memory    
        reconstructed_frame = pad_frame(self._construct_frame((height, width), 128), block_size)
        predicted_frame = pad_frame(self._construct_frame((height, width), 0), block_size)
        reference_frame_buffer = deque([],config.nRefFrames)

        # Start encoding
        for k, frame in enumerate(frames[:10]):
            # Decide I_frame or P_frame
            frame_mode = self._decide_frame_mode(k)
            Y_frame, _, _ = self.yuv_operator.get_YUV_from_frame(frame)
            Y_frame = pad_frame(Y_frame, block_size)
            frame_data = bitarray(1)
            block_data_lengths = bitarray()
            encoded_data = bitarray()
            current_mvp = [0, 0]

            # Decide reference frame
            Y_frame, _, _ = self.yuv_operator.get_YUV_from_frame(frame)
            if frame_mode == 'I':
                # Using current frame
                current_I_mode = config.Default_start_I_mode
                frame_data[0]=0
                current_mv = self.config.Default_start_mv

            elif frame_mode == 'P':
                # Using reconstructed previous frame
                current_mv = config.Default_start_mv
                frame_data[0]=1
            
            # PRE_RUN
            start_time = time.time()   
            frame_mae = 0  
            blocks_counts = 0
            bitrates = 0
            split_map = bitarray()
            split_counts = 0
            array_length = []

            # split to blocks  
            for i in range(0, self.config.height, block_size): 
                for j in range(0, self.config.width, block_size):
                    if j == 0:
                        current_mvp = [0, 0]
                    encoded_block_data = []

                    # multiprocess 
                    block = Block(i, j, block_size, Y_frame)
                    
                    # Predict
                    if self.config.VBSEnable == True and frame_mode == 'P':
                        split, predicted_block, reconstructed_block, encoded_block_data, current_mv, block_data_length, mae = \
                            self.VBS_predict(block, frame_mode, reference_frame_buffer, reconstructed_frame, current_mvp, current_mv, current_I_mode)
                        current_mvp = [current_mv[0], current_mv[1]]
                        if split == True:
                            split_map.extend('1')
                            blocks_counts += 4
                            split_counts += 1
                        else:
                            split_map.extend('0')
                            blocks_counts += 1
                        bitrates += len(encoded_block_data)
                        encoded_data.extend(encoded_block_data)
                        block_data_lengths.extend(block_data_length)
                        frame_mae += mae
                        # with open('mv.txt', 'a+') as f:
                        #     print(current_mv, file=f)
                    elif self.config.VBSEnable == True and frame_mode == 'I':
                        split, predicted_block, reconstructed_block, encoded_block_data, current_I_mode, block_data_length, mae = \
                            self.VBS_predict(block, frame_mode, reference_frame_buffer, reconstructed_frame, current_mvp, current_mv, current_I_mode)
                        current_mvp = [current_mv[0], current_mv[1]]
                        if split == True:
                            split_map.extend('1')
                            blocks_counts += 4
                            split_counts += 1
                        else:
                            split_map.extend('0')
                            blocks_counts += 1
                        bitrates += len(encoded_block_data)
                        encoded_data.extend(encoded_block_data)
                        block_data_lengths.extend(block_data_length)
                        frame_mae += mae
                    else:
                        blocks_counts += 1
                        predicted_block, mae, extra_data = self.predict(block, frame_mode=frame_mode, reference_frame_buffer=reference_frame_buffer, reconstructed_frame=reconstructed_frame, current_mvp = current_mvp)   
                        frame_mae += mae

                        # Transform Residual Block
                        residual_block = Block(block.x, block.y, block_size, None, block.block_data-predicted_block.block_data)
                        coff = self.transform_block(residual_block.block_data, self.config.QP, block_size)
                        # Reconstruct Frame
                        residual_block_data = decoder.inverse_transform_block_data(coff, self.config.QP, block_size)
                        inversed_residual_block = Block(block.x, block.y, block_size, None, residual_block_data)
                        reconstructed_block = self._reconstruct_block_(predicted_block=predicted_block, residual_block=inversed_residual_block)

                        # if frame_mode == 'P':
                        #     with open('mv.txt', 'a+') as f:
                        #         print(extra_data, file=f)

                        if frame_mode == 'I':
                            i_mode_differential = extra_data - current_I_mode
                            current_I_mode = extra_data
                            encoded_block_data.append(i_mode_differential)
                            encoded_block_data.extend(self.entropy_coder.reorder(coff, block_size))
                            encoded_block_data = self.entropy_coder.rle(encoded_block_data)
                            encoded_block_data = self.entropy_coder.golomb_encode_array(encoded_block_data, self.config.golomb_m)
                            block_data_lengths.extend(bitarray(format(len(encoded_block_data), 'b').zfill(16)))
                            encoded_data.extend(encoded_block_data)
                            bitrates += len(encoded_block_data)

                        elif frame_mode == 'P':
                            mv_differential = extra_data - current_mv
                            current_mv = extra_data
                            current_mvp = [extra_data[0], extra_data[1]]
                            encoded_block_data.extend(mv_differential)
                            encoded_block_data.extend(self.entropy_coder.reorder(coff, block_size))
                            encoded_block_data = self.entropy_coder.rle(encoded_block_data)
                            encoded_block_data = self.entropy_coder.golomb_encode_array(encoded_block_data, self.config.golomb_m)
                            block_data_lengths.extend(bitarray(format(len(encoded_block_data), 'b').zfill(16)))
                            encoded_data.extend(encoded_block_data)
                            bitrates += len(encoded_block_data)
                            array_length.append(len(encoded_block_data))

                    # Save to shared memory
                    predicted_frame[i:i+block_size, j:j+block_size] = predicted_block.block_data
                    reconstructed_frame[i:i+block_size, j:j+block_size] = reconstructed_block.block_data
                    

            # POST_RUN
            if frame_mode == 'P':
                if (len(reference_frame_buffer) == reference_frame_buffer.maxlen):
                    reference_frame_buffer.popleft()
                reference_frame_buffer.append(reconstructed_frame.copy())
            elif frame_mode == 'I':
                reference_frame_buffer.clear()
                reference_frame_buffer.append(reconstructed_frame.copy())
            frame_data.extend(bitarray(format(blocks_counts, 'b').zfill(16)))
            if self.config.VBSEnable:
                frame_data.extend(split_map)
            frame_data.extend(block_data_lengths)
            array_lengths = [int(block_data_lengths[i:i+16].to01(), 2) for i in range(0, len(block_data_lengths), 16)]
            # print(array_lengths)
            frame_data.extend(encoded_data)

            endtime = time.time()
            total_bitrates.append(bitrates)
            if self.config.VBSEnable == True and frame_mode == 'P':
                split_percentage.append(split_counts/(blocks_counts-split_counts*3))
            
            # Dump file
            with open(f'{ws}/encoded_data.13', 'ab+') as f:
                f.write(frame_data)

            # if k == 3:
            #     previous_frame = Y_frame.copy()
            # if k == 7:
            #     self.yuv_operator.convert_Y_to_png(np.abs(8*(Y_frame-previous_frame)), f'{ws}/png_sequence/Y_frame_different.png')
            #     print(f'{PSNR(previous_frame, Y_frame)=}')
            # temp output   
            # self.yuv_operator.convert_Y_to_png(Y_frame, f'{ws}/png_sequence/Y_frame{k}.png')    
            # self.yuv_operator.convert_Y_to_png(predicted_frame, f'{ws}/png_sequence/Y_frame_predicted{k}.png') 
            # self.yuv_operator.convert_Y_to_png(np.abs(Y_frame-reference_frame), f'{ws}/png_sequence/Y_frame_diff{k}.png')  
            # self.yuv_operator.convert_Y_to_png(np.abs(Y_frame-predicted_frame), f'{ws}/png_sequence/Y_frame_diff_after{k}.png')
            # self.yuv_operator.convert_Y_to_png(reconstructed_frame, f'{ws}/png_sequence/Y_frame_reconstructed{k}.png') 
            print(f"Frame{k}:\
            Type: {frame_mode},\
            PSNR: {round(PSNR(convert_to_uint8(Y_frame[:height, :width]), convert_to_uint8(reconstructed_frame[:height, :width])), 2)},\
            SSIM: {round(SSIM(convert_to_uint8(Y_frame[:height, :width]), convert_to_uint8(reconstructed_frame[:height, :width])), 2)},\
            MAE: {round(frame_mae/blocks_counts, 2)},\
            Per_Frame_Time: {round(endtime-start_time, 2)},\
            Bitrates: {round(bitrates/(8*1024), 2)},\
            Split_counts: {split_counts}") 
            total_time.append(endtime - start_time)
            psnr.append(PSNR(convert_to_uint8(Y_frame[:height, :width]), convert_to_uint8(reconstructed_frame[:height, :width])))
        
        return psnr, total_bitrates, total_time, split_percentage


if __name__ == '__main__':
    # count_time()
    encoder = Encoder()
    psnr, _, time_spent, _ = encoder.encode('Videos/foreman_420p.yuv')
    # print(psnr)
    # print(time_spent)