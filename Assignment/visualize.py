import time

from bitarray import bitarray
import numpy as np
from scipy.fftpack import dct, idct
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from Block import Block, pad_frame, PSNR, SSIM, convert_to_uint8
import Decoder
import Encoder
from Encoder import Encoder
from Decoder import Decoder
from PIL import Image
from matplotlib import patches
import EntropyCoding
from GlobalConfig import Config
from YUV_formater import YUV_Operator

def rd_curve_block16():
    encoder = Encoder()
    encoder.config.block_size=16
    markder_points = [-11, -8, -5, -2]
    marker_style = ['o', 's', '^', 'x']

    encoder.config.I_Period = 1
    encoder.encode('Videos/foreman_420p.yuv')

    plt.xlabel(f'Bitrates')
    plt.ylabel(f'PSNR')

    plt.title('RD-Plot (i=16)')

    #I_period = 1
    psnr_list1 = []
    total_birates_list1 = []
    for qp in range(10, -1, -1):
        encoder.config.QP = qp
        psnr, total_birates, _= encoder.encode('Videos/foreman_420p.yuv')
        psnr_list1.append(psnr)
        total_birates_list1.append(total_birates)

    #I_period = 4
    encoder.config.I_Period = 4
    psnr_list4 = []
    total_birates_list4 = []
    for qp in range(10, -1, -1):
        encoder.config.QP = qp
        psnr, total_birates, _= encoder.encode('Videos/foreman_420p.yuv')
        psnr_list4.append(psnr)
        total_birates_list4.append(total_birates)

    #I_period = 10
    encoder.config.I_Period = 10
    psnr_list10 = []
    total_birates_list10 = []
    for qp in range(10, -1, -1):
        encoder.config.QP = qp
        psnr, total_birates, _= encoder.encode('Videos/foreman_420p.yuv')
        psnr_list10.append(psnr)
        total_birates_list10.append(total_birates)   

    legend_elements = [
        Line2D([0], [0], color='b', label='I_Period=1', linestyle='-'),
        Line2D([0], [0], color='g', label='I_Period=4', linestyle='--'),
        Line2D([0], [0], color='r', label='I_Period=10', linestyle='-.'),
        Line2D([0], [0], marker='o', label=f'QP=10', color='black'),
        Line2D([0], [0], marker='s', label=f'QP=7', color='black'),
        Line2D([0], [0], marker='^', label=f'QP=4', color='black'),
        Line2D([0], [0], marker='x', label=f'QP=1', color='black'),
    ]
 


    plt.plot(total_birates_list1, psnr_list1, label='I_Period=1', color='b', linestyle='-', )
    plt.plot(total_birates_list4, psnr_list4, label='I_Period=4', color='g', linestyle='--')
    plt.plot(total_birates_list10, psnr_list10, label='I_Period=10', color='r', linestyle='-.')
    for index, i in enumerate(markder_points):
        plt.scatter(total_birates_list1[i], psnr_list1[i], color='b', marker=marker_style[index])
        plt.scatter(total_birates_list4[i], psnr_list4[i], color='g', marker=marker_style[index])
        plt.scatter(total_birates_list4[i], psnr_list10[i], color='r', marker=marker_style[index])

    plt.legend(handles=legend_elements, loc='lower right')
    plt.show()

def rd_curve_block8():
    encoder = Encoder()
    encoder.config.block_size=8
    markder_points = [-10, -7, -4, -1]
    marker_style = ['o', 's', '^', 'x']

    plt.xlabel(f'Bitrates')
    plt.ylabel(f'PSNR')

    encoder.config.I_Period = 1
    encoder.encode('Videos/foreman_420p.yuv')

    plt.title('RD-Plot (i=8)')

    #I_period = 1
    psnr_list1 = []
    total_birates_list1 = []
    for qp in range(10, -1, -1):
        encoder.config.QP = qp
        psnr, total_birates, _= encoder.encode('Videos/foreman_420p.yuv')
        psnr_list1.append(psnr)
        total_birates_list1.append(total_birates)

    #I_period = 4
    encoder.config.I_Period = 4
    psnr_list4 = []
    total_birates_list4 = []
    for qp in range(10, -1, -1):
        encoder.config.QP = qp
        psnr, total_birates, _= encoder.encode('Videos/foreman_420p.yuv')
        psnr_list4.append(psnr)
        total_birates_list4.append(total_birates)

    #I_period = 10
    encoder.config.I_Period = 10
    psnr_list10 = []
    total_birates_list10 = []
    for qp in range(10, -1, -1):
        encoder.config.QP = qp
        psnr, total_birates, _= encoder.encode('Videos/foreman_420p.yuv')
        psnr_list10.append(psnr)
        total_birates_list10.append(total_birates)   

    legend_elements = [
        Line2D([0], [0], color='b', label='I_Period=1', linestyle='-'),
        Line2D([0], [0], color='g', label='I_Period=4', linestyle='--'),
        Line2D([0], [0], color='r', label='I_Period=10', linestyle='-.'),
        Line2D([0], [0], marker='o', label=f'QP=9', color='black'),
        Line2D([0], [0], marker='s', label=f'QP=6', color='black'),
        Line2D([0], [0], marker='^', label=f'QP=3', color='black'),
        Line2D([0], [0], marker='x', label=f'QP=0', color='black'),
    ]
 


    plt.plot(total_birates_list1, psnr_list1, label='I_Period=1', color='b', linestyle='-', )
    plt.plot(total_birates_list4, psnr_list4, label='I_Period=4', color='g', linestyle='--')
    plt.plot(total_birates_list10, psnr_list10, label='I_Period=10', color='r', linestyle='-.')
    for index, i in enumerate(markder_points):
        plt.scatter(total_birates_list1[i], psnr_list1[i], color='b', marker=marker_style[index])
        plt.scatter(total_birates_list4[i], psnr_list4[i], color='g', marker=marker_style[index])
        plt.scatter(total_birates_list4[i], psnr_list10[i], color='r', marker=marker_style[index])

    plt.legend(handles=legend_elements, loc='lower right')
    plt.show()

def count_time():
    encoder = Encoder()
    total_time_list1_8 = []
    total_time_list1_16 = []
    total_time_list4_8 = []
    total_time_list4_16 = []
    total_time_list10_8 = []
    total_time_list10_16 = []

    # I_period = 1
    encoder.config.I_Period = 1
    encoder.config.block_size = 8
    for qp in range(10, -1, -1):
        encoder.config.QP = qp
        _, _, time_spent = encoder.encode('Videos/foreman_420p.yuv')
        total_time_list1_8.append(time_spent)
        print(time_spent)
    encoder.config.block_size = 16
    for qp in range(10, -1, -1):
        encoder.config.QP = qp
        _, _, time_spent = encoder.encode('Videos/foreman_420p.yuv')
        total_time_list1_16.append(time_spent)
        print(time_spent)

    # I_period = 4
    encoder.config.I_Period = 4
    encoder.config.block_size = 8
    for qp in range(10, -1, -1):
        encoder.config.QP = qp
        _, _, time_spent= encoder.encode('Videos/foreman_420p.yuv')
        total_time_list4_8.append(time_spent)
        print(time_spent)
    encoder.config.block_size = 16
    for qp in range(10, -1, -1):
        encoder.config.QP = qp
        _, _, time_spent= encoder.encode('Videos/foreman_420p.yuv')
        total_time_list4_16.append(time_spent)
        print(time_spent)


    # I_period = 10
    encoder.config.I_Period = 10
    encoder.config.block_size = 8
    for qp in range(10, -1, -1):
        encoder.config.QP = qp
        _, _, time_spent= encoder.encode('Videos/foreman_420p.yuv')
        total_time_list10_8.append(time_spent)
        print(time_spent)
    encoder.config.block_size = 16
    for qp in range(10, -1, -1):
        encoder.config.QP = qp
        _, _, time_spent= encoder.encode('Videos/foreman_420p.yuv')
        total_time_list10_16.append(time_spent)
        print(time_spent)

    legend_elements = [
        Line2D([0], [0], color='b', label='I_Period=1, Block_size=8', linestyle='-', marker='o'),
        Line2D([0], [0], color='g', label='I_Period=4, Block_size=8', linestyle='--', marker='s'),
        Line2D([0], [0], color='r', label='I_Period=10, Block_size=8', linestyle='-.', marker='^'),
        Line2D([0], [0], color='b', label='I_Period=1, Block_size=16', linestyle=':', marker='o'),
        Line2D([0], [0], color='g', label='I_Period=4, Block_size=16', linestyle=' ', marker='s'),
        Line2D([0], [0], color='r', label='I_Period=10, Block_size=16', linestyle='solid', marker='^'),
    ]

    qp = [i for i in range(10, -1, -1)]
    plt.title('Excution time(For first 10 frames)')
    plt.xlabel('QP')
    plt.ylabel('Time(seconds)')
    plt.plot(qp, total_time_list1_8, color='b', label='I_Period=1, Block_size=8', linestyle='-', marker='o')
    plt.plot(qp, total_time_list1_16, color='b', label='I_Period=1, Block_size=16', linestyle=':', marker='o')
    plt.plot(qp, total_time_list4_8, color='g', label='I_Period=4, Block_size=8', linestyle='-', marker='s')
    plt.plot(qp, total_time_list4_16, color='g', label='I_Period=4, Block_size=16', linestyle=':', marker='s')
    plt.plot(qp, total_time_list10_8, color='r', label='I_Period=10, Block_size=8', linestyle='-', marker='^')
    plt.plot(qp, total_time_list10_16, color='r', label='I_Period=10, Block_size=16', linestyle=':', marker='^')
    plt.legend(handles=legend_elements, loc='upper right')
    plt.show()

def P_I_frame_visualization(image_path: str, block_data: list[list], save_path: str):
    # block_data is like [(x, y, block_width, block_height), .....]
    # Load the original image
    image = Image.open(image_path)
    fig, ax = plt.subplots(1)
    ax.imshow(image, cmap="gray")

    # Get the block size data for specific frame
    for [x, y, block_width, block_height] in block_data:
        # Create a Rectangle patch
        rect = patches.Rectangle((y,x), block_width, block_height, linewidth=1, edgecolor='black', facecolor='none')
        # Add the patch to the Axes
        ax.add_patch(rect)

    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

def reference_frame_visualization(image_path: str, ref_frame_data: list[list], save_path: str):
    # block_data is like [(x, y, block_size, reference_frame_index), .....]
    # Load the original image
    image = Image.open(image_path)
    fig, ax = plt.subplots(1)
    ax.imshow(image, cmap="gray")

    # Define colors for different reference frames
    colors = ['red', 'green', 'blue', 'cyan', 'magenta', 'yellow', 'black', 'white']

    # Overlay reference frame indices on the image
    for [x, y, block_size, reference_frame_index] in ref_frame_data:
        # Choose a color based on the reference frame index
        color = colors[reference_frame_index % len(colors)]
        # Create a Rectangle patch
        rect = patches.Rectangle((y, x), block_size, block_size, linewidth=1, edgecolor='none', facecolor=color, alpha=0.3)
        # Add the patch to the Axes
        ax.add_patch(rect)

    plt.axis("off")
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

def motion_vector_visualization(image_path: str, P_frame_data: list[list], save_path: str, FMEE=False):
    # block_data is like [(x, y, block_size, reference_frame_index), .....]
    # Load the original image
    image = Image.open(image_path)
    fig, ax = plt.subplots()
    ax.imshow(image, cmap='gray', interpolation='none')

    for x, y, mv_x, mv_y, block_size in P_frame_data:
        start_x, start_y = np.flipud([x+block_size//2, y+block_size//2])
        dx, dy = np.flipud([mv_x, mv_y])
        if FMEE:
            dx //= 2
            dy //= 2
        ax.arrow(start_x, start_y, dx, dy, head_length=2, head_width=4, fc='black', ec='black', length_includes_head=True, linewidth=0.5)
    
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

def main():
    config = Config()
    encoder = Encoder()
    psnr, _, time_spent, _ = encoder.encode('Videos/foreman_420p.yuv')
    decoder = Decoder(config)
    frame_visualize_data = decoder.decode()
    print(len(frame_visualize_data))
    # P-I frame visualization
    image_path = "Assignment/decoder_ws/png_sequence/reconstructed_frame"+str(4)+".png"
    save_path = "Assignment/P_I_frame_visualization/frame"+str(4)+".png"
    P_I_frame_visualization(image_path, frame_visualize_data[4][1], save_path)
    # Reference frame visualization
    save_path = "Assignment/reference_frame_visualization/frame"+str(4)+".png"
    reference_frame_visualization(image_path, frame_visualize_data[4][0], save_path)
    # Motion vector visualization
    save_path = "Assignment/motion_vector_visualization/frame"+str(4)+".png"
    motion_vector_visualization(image_path, frame_visualize_data[4][2], save_path)

if __name__ == "__main__":
    main()