from bitarray import bitarray
import math
import numpy as np

from GlobalConfig import Config
from Block import Block
config = Config()

class EntropyCoder:

    def __init__(self) -> None:
        self.config = config

    
    def reorder(self, block_data:np.ndarray, block_size):
        reordered_array = []
        for d in range(2*block_size-1):
            for i in range(max(0, d - block_size + 1), min(d, block_size - 1) + 1):
                j = d - i
                reordered_array.append(block_data[i, j])
        return reordered_array

    def inverse_reorder(self, array:np.ndarray, block_size):
        block_data = np.zeros((block_size, block_size), dtype=np.int16)
        index = 0
        for d in range(2*block_size-1):
            for i in range(max(0, d - block_size + 1), min(d, block_size - 1) + 1):
                j = d - i
                block_data[i][j] = array[index]
                index+=1
        return block_data

    # # todo: optimized rle
    def rle(self, array):
        counts = 0
        zero = (array[0] == 0)
        encoded_array = []
        for i, value in enumerate(array):
            if value != 0 and zero == True:
                encoded_array.append(counts)
                counts = 0
                zero = False
            elif value == 0 and zero == False:
                encoded_array.append(-counts)
                encoded_array.extend(array[i-counts:i])
                counts = 0   
                zero = True
            counts += 1
        if zero == True:
            encoded_array.append(0)
        elif zero == False:
            encoded_array.append(-counts)
            encoded_array.extend(array[-counts:])     
        return encoded_array
    

    def inverse_rle(self, encoded_array, array_len):
        decoded_array = [0 for i in range(array_len)]
        i = 0
        j = 0
        while i < len(encoded_array):
            if encoded_array[i] < 0:
                decoded_array[j:j-encoded_array[i]] = encoded_array[i+1:i+1-encoded_array[i]]
                j -= encoded_array[i]
                i -= encoded_array[i]-1
            elif encoded_array[i] > 0:
                j += encoded_array[i]
                i += 1
            elif encoded_array[i] == 0:
                break
        return decoded_array


    def golomb_encoding(self, value, m):
        if value < 0:
            sign = bitarray('1')
            value = -value
        else:
            sign = bitarray('0')
        quotient = value // m
        remainder = value % m
        unary_code = bitarray('1' * quotient + '0')
        binary_code = bitarray(format(remainder, 'b').zfill(int(math.log2(m))))
        return sign + unary_code + binary_code


    def golomb_encode_array(self, array, m):
        encoded_data = bitarray()
        for value in array:
            encoded_data.extend(self.golomb_encoding(value, m))
        return encoded_data


    def golomb_decode_array(self, encoded_data, m):
        decoded_data = []
        index = 0
        while index < len(encoded_data):
            sign = encoded_data[index]
            index += 1
            quotient_length = encoded_data[index:].index(0) + 1
            quotient = quotient_length - 1
            index += quotient_length
            binary_code = encoded_data[index:index + int(math.log2(m))]
            value = quotient * m + int(binary_code.to01(), 2)
            if sign:
                value = -value
            decoded_data.append(value)
            index += int(math.log2(m))
        return decoded_data

    

if __name__ == '__main__':
    coder = EntropyCoder()
    block = Block(0, 0, 3, np.array([[1, 2, 3],[4, 5, 6], [7, 8, 9]]))
    reordered = coder.reorder(block.block_data,3)
    # print(coder._inverse_reorder(reordered, 3))
    array = [0, 0, 123, -67, 8, 0, 0, 4, 0, 0, 0, 2, 0, 2, 2, 3, 0, 0]
    encoded_array = coder.optimized_rle(array)
    print(encoded_array)
    decoded_array = coder.inverse_optimized_rle(encoded_array, 18)
    print(decoded_array)
    golob = coder.golomb_encode_array(array, m=4)
    print(golob)
    decoded_golob = coder.golomb_decode_array(golob, m=4)
    print(decoded_golob)


