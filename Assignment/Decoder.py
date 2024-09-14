import math

from bitarray import bitarray
from collections import deque
import numpy as np
from scipy.fftpack import idct

from Block import pad_frame, Block
import EntropyCoding
from GlobalConfig import Config
from YUV_formater import YUV_Operator

config = Config()

class Decoder:
    def __init__(self, config:Config) -> None:
        self.config = config
        self.yuv_operator = YUV_Operator(config)
        self.entropy_coder = EntropyCoding.EntropyCoder()
        self.current_mv = np.ndarray([0, 0])


    def _construct_virtual_reference_frame(self, frame_shape, value):
        virtual_frame = np.full(frame_shape, value, dtype=np.int16)
        return virtual_frame


    def _reconstruct_block_(self, predicted_block:Block, residual_block:Block):
        return Block(residual_block.x, residual_block.y, residual_block.block_size, \
                     None, (predicted_block.block_data+residual_block.block_data).clip(0, 255))


    def idct_transform(self, block_data):
        return np.round(idct(idct(block_data, axis=0, norm='ortho'), axis=1, norm='ortho')).astype(np.int16)
    

    def dequantize(self, quantized_coefficients:np.ndarray, QP, i):
        Q = self.calculate_quantization_matrix(QP, i)
        return (quantized_coefficients * Q).astype(np.int16)
    

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
    

    def inverse_transform_block_data(self, quantized_coefficients, QP, block_size):
        # 1. 对每个块执行反量化操作
        dct_coefficients = self.dequantize(quantized_coefficients, QP, block_size)
        # 2. 对每个块执行IDCT逆变换
        block_data = self.idct_transform(dct_coefficients)
        return block_data
    

    def decode_predicted_block(self, reference_frame_buffer, block_size, current_mv, i, j):
        if self.FMEEnable == True:
            if current_mv[0] % 2 == 0 and current_mv[1] % 2 == 0:
                predicted_block = Block(i+current_mv[0]//2, j+current_mv[1]//2, block_size, reference_frame_buffer[current_mv[2]])
            elif current_mv[0] % 2 != 0 and current_mv[1] % 2 == 0:
                new_data1 = Block(i+(current_mv[0]+1)//2, j+current_mv[1]//2, block_size, reference_frame_buffer[current_mv[2]]).block_data
                new_data2 = Block(i+(current_mv[0]-1)//2, j+current_mv[1]//2, block_size, reference_frame_buffer[current_mv[2]]).block_data
                predicted_block = Block(i+current_mv[0], j+current_mv[1], block_size, None, (new_data1+new_data2)//2)
            elif current_mv[0] % 2 == 0 and current_mv[1] % 2 != 0:
                new_data1 = Block(i+current_mv[0]//2, j+(current_mv[1]+1)//2, block_size, reference_frame_buffer[current_mv[2]]).block_data
                new_data2 = Block(i+current_mv[0]//2, j+(current_mv[1]-1)//2, block_size, reference_frame_buffer[current_mv[2]]).block_data
                predicted_block = Block(i+current_mv[0], j+current_mv[1], block_size, None, (new_data1+new_data2)//2)
            elif current_mv[0] % 2 != 0 and current_mv[1] % 2 != 0:
                new_data1 = Block(i+(current_mv[0]-1)//2, j+(current_mv[1]+1)//2, block_size, reference_frame_buffer[current_mv[2]]).block_data
                new_data2 = Block(i+(current_mv[0]-1)//2, j+(current_mv[1]-1)//2, block_size, reference_frame_buffer[current_mv[2]]).block_data
                new_data3 = Block(i+(current_mv[0]+1)//2, j+(current_mv[1]-1)//2, block_size, reference_frame_buffer[current_mv[2]]).block_data
                new_data4 = Block(i+(current_mv[0]+1)//2, j+(current_mv[1]+1)//2, block_size, reference_frame_buffer[current_mv[2]]).block_data
                predicted_block = Block(i+current_mv[0], j+current_mv[1], block_size, None, (new_data1+new_data2+new_data3+new_data4)//4)
        else:
            predicted_block = Block(i+current_mv[0], j+current_mv[1], block_size, reference_frame_buffer[current_mv[2]])

        return predicted_block

    def decode_P_frame(self, file:bitarray, start, reference_frame_buffer):
        multi_frame_list = []
        block_size_list = []
        mv_list = []
        blocks_counts = int(file[start:start+16].to01(), 2)
        start += 16
        if self.VBSEnable:
            split_map = file[start:start+self.blocks_counts]
            start += self.blocks_counts
        encoded_block_lengths_data = file[start:start+blocks_counts*16]
        array_lengths = [int(encoded_block_lengths_data[i:i+16].to01(), 2) for i in range(0, len(encoded_block_lengths_data), 16)]
        reconstructed_frame = pad_frame(self._construct_virtual_reference_frame((self.height, self.width), 0), self.block_size)
        mv_diffs = []
        residual_blocks = []
        start+=blocks_counts*16
        blocks_counts = 0
        index = 0

        for i in range(self.blocks_counts):
            if self.VBSEnable == True and split_map[i] == True:
                for j in range(4):
                    array_length = array_lengths[index]
                    encoded_data = bitarray()
                    encoded_data = file[start:start+array_length]
                    decoded_array = self.entropy_coder.golomb_decode_array(encoded_data, self.golomb_m)
                    start+=array_length
                    new_block_size = self.block_size//2
                    if self.qp == 0:
                        new_qp = 0
                    else:
                        new_qp = self.qp-1
                    inversed_rle = self.entropy_coder.inverse_rle(decoded_array, new_block_size**2+3)
                    mv_diff = np.array(inversed_rle[0:3], dtype=np.int8)
                    mv_diffs.append(mv_diff)
                    reordered_block_data = inversed_rle[3:]
                    transformed_block_data = self.entropy_coder.inverse_reorder(reordered_block_data, new_block_size)
                    block_data = self.inverse_transform_block_data(transformed_block_data, new_qp, new_block_size)
                    residual_blocks.append(block_data)
                    index+=1
            else:
                array_length = array_lengths[index]
                encoded_data = bitarray()
                encoded_data = file[start:start+array_length]
                decoded_array = self.entropy_coder.golomb_decode_array(encoded_data, self.golomb_m)
                start+=array_length
                new_block_size = self.block_size
                new_qp = self.qp
                inversed_rle = self.entropy_coder.inverse_rle(decoded_array, new_block_size**2+3)
                mv_diff = np.array(inversed_rle[0:3], dtype=np.int8)
                mv_diffs.append(mv_diff)
                reordered_block_data = inversed_rle[3:]
                transformed_block_data = self.entropy_coder.inverse_reorder(reordered_block_data, new_block_size)
                block_data = self.inverse_transform_block_data(transformed_block_data, new_qp, new_block_size)
                residual_blocks.append(block_data)
                index += 1

        index = 0
        current_mv = np.array([0, 0, 0])
        for i in range(0, self.height, self.block_size): 
            for j in range(0, self.width, self.block_size):
                if self.VBSEnable == True and split_map[blocks_counts] == True:
                    new_block_size = self.block_size//2
                    steps = [0, new_block_size]
                    for x in steps:
                        for y in steps:
                            current_mv = current_mv + mv_diffs[index]
                            residual_block = Block(i, j, new_block_size, None, residual_blocks[index])
                            predicted_block = self.decode_predicted_block(reference_frame_buffer, new_block_size, current_mv, i+x, j+y)
                            reconstructed_block = self._reconstruct_block_(predicted_block=predicted_block, residual_block=residual_block)
                            reconstructed_frame[i+x:i+x+new_block_size, j+y:j+y+new_block_size] = reconstructed_block.block_data
                            index+=1
                            multi_frame_list.append([i+x, j+y, new_block_size, current_mv[2]])    
                            block_size_list.append([i+x, j+y, new_block_size, new_block_size])    
                            mv_list.append([i+x, j+y, current_mv[0], current_mv[1], new_block_size])    
                    blocks_counts+=1
                else:
                    current_mv = current_mv + mv_diffs[index]
                    # with open('predicted_block_decode.txt', 'a+') as f:
                    #     print(current_mv, file=f)  
                    predicted_block = self.decode_predicted_block(reference_frame_buffer, self.block_size, current_mv, i, j)
                    residual_block = Block(i, j, self.block_size, None, residual_blocks[index])
                    # with open('predicted_block_decode.txt', 'a+') as f:
                    #     print(str(residual_block.block_data), file=f)
                    reconstructed_block = self._reconstruct_block_(predicted_block=predicted_block, residual_block=residual_block)
                    reconstructed_frame[i:i+self.block_size, j:j+self.block_size] = reconstructed_block.block_data
                    blocks_counts+=1
                    index += 1
                    multi_frame_list.append([i, j, self.block_size, current_mv[2]])    
                    block_size_list.append([i, j, self.block_size, self.block_size])    
                    mv_list.append([i, j, current_mv[0], current_mv[1], self.block_size])    

        return reconstructed_frame, start, multi_frame_list, block_size_list, mv_list


    def decode_I_frame(self, buffer, start, reference_frame):
        multi_frame_list = []
        block_size_list = []
        mv_list = []
        blocks_counts = int(buffer[start:start+16].to01(), 2)
        start+=16
        if self.VBSEnable:
            split_map = buffer[start:start+self.blocks_counts]
            start += self.blocks_counts
        encoded_block_lengths_data = buffer[start:start+blocks_counts*16]
        array_lengths = [int(encoded_block_lengths_data[i:i+16].to01(), 2) for i in range(0, len(encoded_block_lengths_data), 16)]
        reconstructed_frame = np.zeros(reference_frame.shape, dtype=np.int16)
        I_mode_diffs = []
        residual_blocks = []
        start+=blocks_counts*16
        blocks_counts = 0
        index = 0
        for i in range(self.blocks_counts):
            if self.VBSEnable == True and split_map[i] == True:
                for j in range(4):
                    array_length = array_lengths[index]
                    encoded_data = bitarray()
                    encoded_data = buffer[start:start+array_length]
                    decoded_array = self.entropy_coder.golomb_decode_array(encoded_data, self.golomb_m)
                    start+=array_length
                    new_block_size = self.block_size//2
                    if self.qp == 0:
                        new_qp = 0
                    else:
                        new_qp = self.qp-1
                    inversed_rle = self.entropy_coder.inverse_rle(decoded_array, new_block_size**2+3)
                    I_mode_diff = np.array(inversed_rle[0:1], dtype=np.int8)
                    I_mode_diffs.append(I_mode_diff)
                    reordered_block_data = inversed_rle[1:]
                    transformed_block_data = self.entropy_coder.inverse_reorder(reordered_block_data, new_block_size)
                    block_data = self.inverse_transform_block_data(transformed_block_data, new_qp, new_block_size)
                    residual_blocks.append(block_data)
                    index+=1
            else:
                # Entropy Decoding
                array_length = array_lengths[index]
                encoded_data = bitarray()
                encoded_data = buffer[start:start+array_length]
                decoded_array = self.entropy_coder.golomb_decode_array(encoded_data, self.golomb_m)
                start+=array_length
                inversed_rle = self.entropy_coder.inverse_rle(decoded_array, self.block_size**2+2)
                I_mode_diff = np.array(inversed_rle[0:1], dtype=np.int8)
                I_mode_diffs.append(I_mode_diff)
                reordered_block_data = inversed_rle[1:]
                transformed_block_data = self.entropy_coder.inverse_reorder(reordered_block_data, self.block_size)
                block_data = self.inverse_transform_block_data(transformed_block_data, self.qp, self.block_size)
                residual_blocks.append(block_data)
                index += 1

        current_i_mode = 0
        index = 0
        for i in range(0, self.height, self.block_size): 
            for j in range(0, self.width, self.block_size):
                if self.VBSEnable == True and split_map[blocks_counts] == True:
                    new_block_size = self.block_size//2
                    steps = [0, new_block_size]
                    for x in steps:
                        for y in steps:
                            current_i_mode = current_i_mode + I_mode_diffs[index]
                            if current_i_mode == 1:
                                if i+x == 0:
                                    top_border = np.full((1, new_block_size), 128, dtype=np.int16)
                                else:
                                    top_border = reconstructed_frame[i+x-1:i+x,j+y:j+y+new_block_size]
                                predicted_block_data = np.tile(top_border, (new_block_size, 1))
                                predicted_block = Block(i+x-1, j+y, new_block_size, None, predicted_block_data)
                            elif current_i_mode == 0:
                                if j+y == 0:
                                    left_border = np.full((new_block_size, 1), 128, dtype=np.int16)  # 使用128填充缺失的样本
                                else:
                                    left_border = reconstructed_frame[i+x:i+x+new_block_size,j+y-1:j+y]
                                predicted_block_data = np.tile(left_border, (1, new_block_size))
                                predicted_block = Block(i+x, j+y-1, new_block_size, None, predicted_block_data)
                            
                            # with open('residual_block_decode', 'a+') as f:
                            #     print(str(predicted_block.block_data), file=f)
                            residual_block = Block(i, j, new_block_size, None, residual_blocks[index])
                            # with open('predicted_block_decode.txt', 'a+') as f:
                            #     print(str(residual_block.block_data), file=f)
                            reconstructed_block = self._reconstruct_block_(predicted_block=predicted_block, residual_block=residual_block)
                            reconstructed_frame[i+x:i+x+new_block_size, j+y:j+y+new_block_size] = reconstructed_block.block_data
                            index += 1
                            if current_i_mode == 0:
                                mv_vector=[0, -1]
                            elif current_i_mode == 1:
                                mv_vector = [-1, 0]
                            multi_frame_list.append([i+x, j+y, new_block_size, 0])    
                            block_size_list.append([i+x, j+y, new_block_size, new_block_size])    
                            mv_list.append([i+x, j+y, mv_vector[0],mv_vector[1], new_block_size])    
                    blocks_counts+=1
                else:
                    current_i_mode = current_i_mode + I_mode_diffs[index]
                    if current_i_mode == 1:
                        if i == 0:
                            top_border = np.full((1, self.block_size), 128, dtype=np.int16)
                        else:
                            top_border = reconstructed_frame[i-1:i,j:j+self.block_size]
                        predicted_block_data = np.tile(top_border, (self.block_size, 1))
                        predicted_block = Block(i-1, j, self.block_size, None, predicted_block_data)
                    elif current_i_mode == 0:
                        if j == 0:
                            left_border = np.full((self.block_size, 1), 128, dtype=np.int16)  # 使用128填充缺失的样本
                        else:
                            left_border = reconstructed_frame[i:i+self.block_size,j-1:j]
                        predicted_block_data = np.tile(left_border, (1, self.block_size))
                        predicted_block = Block(i, j-1, self.block_size, None, predicted_block_data)
                    
                    # with open('residual_block_decode', 'a+') as f:
                    #     print(str(predicted_block.block_data), file=f)
                    residual_block = Block(i, j, self.block_size, None, residual_blocks[index])
                    # with open('predicted_block_decode.txt', 'a+') as f:
                    #     print(str(residual_block.block_data), file=f)
                    reconstructed_block = self._reconstruct_block_(predicted_block=predicted_block, residual_block=residual_block)
                    reconstructed_frame[i:i+self.block_size, j:j+self.block_size] = reconstructed_block.block_data
                    blocks_counts+=1
                    index += 1
                    if current_i_mode == 0:
                        mv_vector=[0, -1]
                    elif current_i_mode == 1:
                        mv_vector = [-1, 0]
                    multi_frame_list.append([i, j, self.block_size, 0])    
                    block_size_list.append([i, j, self.block_size, self.block_size])    
                    mv_list.append([i, j, mv_vector[0],mv_vector[1], self.block_size]) 

        return reconstructed_frame, start, multi_frame_list, block_size_list, mv_list
    


    def decode(self, input_dir = 'Assignment/encoder_ws' ,ws = 'Assignment/decoder_ws'):
        buffer = bitarray()
        frame_counts = 0

        # header按字节顺序依次为
        # 2B blocks_counts
        # 2B width
        # 2B height
        # 1B block_size
        # 1B QP
        # 1B golomb_m 
        with open(f'{input_dir}/encoded_data.13', 'rb') as file:
            #todo读取header
            buffer.fromfile(file)
            start = 0
            self.blocks_counts = int(buffer[start:start+16].to01(), 2)
            self.width = int(buffer[start+16:start+32].to01(), 2)
            self.height = int(buffer[start+32:start+48].to01(), 2)
            self.block_size = int(buffer[start+48:start+56].to01(), 2)
            self.qp = int(buffer[start+56:start+64].to01(), 2)
            self.golomb_m = int(buffer[start+64:start+72].to01(), 2)
            self.nrefFrames = int(buffer[start+72:start+77].to01(), 2)
            self.VBSEnable = bool(buffer[start+77])
            self.FMEEnable = bool(buffer[start+78])
            self.fastME = bool(buffer[start+79])

        reference_frame_buffer = deque([],self.nrefFrames)
        reference_frame = self._construct_virtual_reference_frame((self.height, self.width), 128)
        reference_frame = pad_frame(reference_frame, self.block_size)
        reconstructed_frame = self._construct_virtual_reference_frame((self.height, self.width), 128)
        reconstructed_frame = pad_frame(reference_frame, self.block_size)
        start = 80
        frame_visualize_data = []
        print(len(buffer))
        while start < len(buffer)-8:
            # 字节对齐
            start = math.ceil(start/8)*8
            mode = int(buffer[start:start+1].to01(), 2)
            start+=1
            frame_counts += 1
            if mode == 1:
                reference_frame = reconstructed_frame
                reconstructed_frame, start, mul, block_size_list, mv_list = self.decode_P_frame(buffer, start, reference_frame_buffer)
                if (len(reference_frame_buffer) == reference_frame_buffer.maxlen):
                    reference_frame_buffer.popleft()
                reference_frame_buffer.append(reconstructed_frame.copy())          
                self.yuv_operator.convert_Y_to_png(reconstructed_frame, f"{ws}/png_sequence/reconstructed_frame{frame_counts-1}.png")
                frame_visualize_data.append([mul, block_size_list, mv_list])
            elif mode == 0:
                reference_frame = self._construct_virtual_reference_frame((self.height, self.width), 128)
                reference_frame = pad_frame(reference_frame, self.block_size)
                reconstructed_frame, start, mul, block_size_list, mv_list = self.decode_I_frame(buffer, start, reference_frame)
                reference_frame_buffer.clear()
                reference_frame_buffer.append(reconstructed_frame.copy())
                self.yuv_operator.convert_Y_to_png(reconstructed_frame, f"{ws}/png_sequence/reconstructed_frame{frame_counts-1}.png")          
                frame_visualize_data.append([mul, block_size_list, mv_list])
                
        return frame_visualize_data

if __name__ == '__main__':
    decoder = Decoder(config)
    frame_visualize_data = decoder.decode()
