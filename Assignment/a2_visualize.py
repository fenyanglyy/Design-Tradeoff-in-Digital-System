import numpy as np
import matplotlib.pyplot as plt

import Encoder


def rd_curve_a2():
    encoder = Encoder.Encoder()
    
    # fixed parameters
    encoder.config.block_size = 16
    encoder.config.search_range = 4
    encoder.config.I_Period = 8

    # original encoder
    encoder.config.nRefFrames = 1
    encoder.config.VBSEnable = False
    encoder.config.FMEEnable = False
    encoder.config.FastME = False
    psnr_list1 = []
    total_birates_list1 = []
    total_time_list1 = []
    for qp in [i for i in range(1, 11)]:
        encoder.config.QP = qp
        psnr, total_birates, time_spent, _= encoder.encode('Videos/foreman_420p.yuv')
        psnr_list1.append(np.mean(psnr))
        total_birates_list1.append(np.sum(total_birates))
        total_time_list1.append(np.sum(time_spent))

    # # multi-ref encoder
    encoder.config.nRefFrames = 4
    encoder.config.VBSEnable = False
    encoder.config.FMEEnable = False
    encoder.config.FastME = False
    psnr_list2 = []
    total_birates_list2 = []
    total_time_list2 = []
    for qp in [i for i in range(1, 11)]:
        encoder.config.QP = qp
        psnr, total_birates, time_spent, _= encoder.encode('Videos/foreman_420p.yuv')
        psnr_list2.append(np.mean(psnr))
        total_birates_list2.append(np.sum(total_birates))
        total_time_list2.append(np.sum(time_spent))

    # vbs encoder
    encoder.config.nRefFrames = 1
    encoder.config.VBSEnable = True
    encoder.config.FMEEnable = False
    encoder.config.FastME = False
    psnr_list3 = []
    total_birates_list3 = []
    total_time_list3 = []
    for qp in [i for i in range(1, 11)]:
        encoder.config.QP = qp
        psnr, total_birates, time_spent, _= encoder.encode('Videos/foreman_420p.yuv')
        psnr_list3.append(np.mean(psnr))
        total_birates_list3.append(np.sum(total_birates))
        total_time_list3.append(np.sum(time_spent))

    # # fmee encoder
    encoder.config.nRefFrames = 1
    encoder.config.VBSEnable = False
    encoder.config.FMEEnable = True
    encoder.config.FastME = False
    psnr_list4 = []
    total_birates_list4 = []
    total_time_list4 = []

    for qp in [i for i in range(1, 11)]:
        encoder.config.QP = qp
        psnr, total_birates, time_spent, _= encoder.encode('Videos/foreman_420p.yuv')
        psnr_list4.append(np.mean(psnr))
        total_birates_list4.append(np.sum(total_birates))
        total_time_list4.append(np.sum(time_spent))

    # # fmee encoder
    encoder.config.nRefFrames = 1
    encoder.config.VBSEnable = False
    encoder.config.FMEEnable = False
    encoder.config.FastME = True
    psnr_list5 = []
    total_birates_list5 = []
    total_time_list5 = []
    for qp in [i for i in range(1, 11)]:
        encoder.config.QP = qp
        psnr, total_birates, time_spent, _= encoder.encode('Videos/foreman_420p.yuv')
        psnr_list5.append(np.mean(psnr))
        total_birates_list5.append(np.sum(total_birates))
        total_time_list5.append(np.sum(time_spent))


    # allfeature encoder
    encoder.config.nRefFrames = 4
    encoder.config.VBSEnable = True
    encoder.config.FMEEnable = True
    encoder.config.FastME = True
    psnr_list6 = []
    total_birates_list6 = []
    total_time_list6 = []
    for qp in [i for i in range(1, 11)]:
        encoder.config.QP = qp
        psnr, total_birates, time_spent, _= encoder.encode('Videos/foreman_420p.yuv')
        psnr_list6.append(np.mean(psnr))
        total_birates_list6.append(np.sum(total_birates))
        total_time_list6.append(np.sum(time_spent))

    fig, (ax1, ax2) = plt.subplots(1, 2)
    
    ax1.plot(total_birates_list1, psnr_list1, label='Original', linestyle='-', color='blue', marker='o')
    ax1.plot(total_birates_list2, psnr_list2, label='MultiRef', linestyle='--', color='green', marker='s')
    ax1.plot(total_birates_list3, psnr_list3, label='VBSEnable', linestyle='-.', color='red', marker='x')
    ax1.plot(total_birates_list4, psnr_list4, label='FMEEnable', linestyle='-', color='purple', marker='o')
    ax1.plot(total_birates_list5, psnr_list5, label='FastME', linestyle='--', color='orange', marker='s')
    ax1.plot(total_birates_list6, psnr_list6, label='ALLFEATURE', linestyle='-.', color='gray', marker='x')
    ax1.set_xlabel('Bitrates')
    ax1.set_ylabel('PSNR')
    ax1.set_title("RD_curve of Prediction Features")
    ax1.legend()


    qp = [i for i in range(1, 11)]
    ax2.plot(qp, total_time_list1, label='Original', linestyle='-', color='blue', marker='o')
    ax2.plot(qp, total_time_list2, label='MultiRef', linestyle='--', color='green', marker='s')
    ax2.plot(qp, total_time_list3, label='VBSEnable', linestyle='-.', color='red', marker='x')
    ax2.plot(qp, total_time_list4, label='FMEEnable', linestyle='-', color='purple', marker='o')
    ax2.plot(qp, total_time_list5, label='FastME', linestyle='--', color='orange', marker='s')
    ax2.plot(qp, total_time_list6, label='ALLFEATURE', linestyle='-.', color='gray', marker='x')
    ax2.set_xlabel('QP')
    ax2.set_ylabel('Time(s)')
    ax2.set_title("Time Spent of Prediction Features")
    ax2.legend()

    plt.tight_layout()
    # plt.show()
    plt.savefig(f'Assignment/rd_curve/rdcurve.png')

def a2_split_percentage():
    encoder = Encoder.Encoder()
    
    # fixed parameters
    encoder.config.block_size = 16
    encoder.config.search_range = 4
    encoder.config.I_Period = 8
    
    # vbs encoder
    encoder.config.nRefFrames = 1
    encoder.config.VBSEnable = True
    encoder.config.FMEEnable = False
    encoder.config.FastME = False
    psnr_list3 = []
    total_birates_list3 = []
    total_time_list3 = []
    split_percentage_list3 = []
    for qp in [i for i in range(1, 11)]:
        encoder.config.QP = qp
        psnr, total_birates, time_spent, split_percentage= encoder.encode('Videos/foreman_420p.yuv')
        psnr_list3.append(np.mean(psnr))
        total_birates_list3.append(np.sum(total_birates))
        total_time_list3.append(np.sum(time_spent))
        split_percentage_list3.append(np.mean(split_percentage))

    fig, (ax1, ax2) = plt.subplots(2, 1)
    qp = [i for i in range(1, 11)]
    ax1.plot(qp, split_percentage_list3, label='VBSEnable', linestyle='-', color='blue', marker='o')
    ax1.set_xlabel('QP')
    ax1.set_ylabel('Split Blocks Percentage')
    ax2.plot(total_birates_list3, split_percentage_list3, label='VBSEnable', linestyle='-', color='blue', marker='o')
    ax2.set_xlabel('Bitrates')
    ax2.set_ylabel('Split Blocks Percentage')
    ax1.legend()
    ax2.legend()

    plt.tight_layout()
    # plt.show()
    plt.savefig(f'Assignment/rd_curve/percentage.png')


def multi_ref_psnr():
    encoder = Encoder.Encoder()
    
    # fixed parameters
    encoder.config.block_size = 16
    encoder.config.search_range = 4
    encoder.config.I_Period = 10
    encoder.config.VBSEnable = False
    encoder.config.FMEEnable = False
    encoder.config.FastME = False
    encoder.config.QP = 4
    # multi-ref encoder
    encoder.config.nRefFrames = 1
    psnr1, total_birates1, _, _= encoder.encode('Videos/synthetic.yuv')

    # multi-ref encoder
    encoder.config.nRefFrames = 2
    psnr2, total_birates2, _, _= encoder.encode('Videos/synthetic.yuv')
    
    # multi-ref encoder
    encoder.config.nRefFrames = 3
    psnr3, total_birates3, _, _= encoder.encode('Videos/synthetic.yuv')

    # multi-ref encoder
    encoder.config.nRefFrames = 4
    psnr4, total_birates4, _, _= encoder.encode('Videos/synthetic.yuv')

    fig, (ax1, ax2) = plt.subplots(2, 1)

    x = [i for i in range(10)]

    ax1.plot(x, psnr1, label='nRefFrames=1', linestyle='-', color='blue', marker='o')
    ax1.plot(x, psnr2, label='nRefFrames=2', linestyle='--', color='green', marker='s')
    ax1.plot(x, psnr3, label='nRefFrames=3', linestyle='-.', color='red', marker='x')
    ax1.plot(x, psnr4, label='nRefFrames=4', linestyle=':', color='purple', marker='^')

    ax2.plot(x, total_birates1, label='nRefFrames=1', linestyle='-', color='blue', marker='o')
    ax2.plot(x, total_birates2, label='nRefFrames=2', linestyle='--', color='green', marker='s')
    ax2.plot(x, total_birates3, label='nRefFrames=3', linestyle='-.', color='red', marker='x')
    ax2.plot(x, total_birates4, label='nRefFrames=4', linestyle=':', color='purple', marker='^')

    ax1.set_xlabel('Frame Index')
    ax1.set_ylabel('PSNR')
    ax1.set_title('MultiRef Per Frame PSNR')

    ax2.set_xlabel('Frame Index')
    ax2.set_ylabel('Bitrates')
    ax2.set_title('MultiRef Per Frame Bitrates')

    ax1.legend()
    ax2.legend()
    plt.tight_layout()
    # plt.show()
    plt.savefig(f'Assignment/rd_curve/per_frame_psnr.png')


if __name__ == '__main__':
    # start_index = sys.argv[1]
    # for i in range(int(start_index), int(start_index)+50, 5):
    #     constant = i/1000
    #     rd_curve_a2(constant)
    # a2_split_percentage()
    rd_curve_a2()
    # a2_split_percentage()
    # multi_ref_psnr()

    