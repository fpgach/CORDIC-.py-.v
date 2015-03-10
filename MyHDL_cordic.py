import math as math
import numpy as np
from myhdl import *
import matplotlib.pyplot as plt

def Divider(i_clk, i_rst, i_en, i_divident, i_divider, o_quotient, o_reminder):
    n = max(len(i_divident), len(i_divider))
    M = 2*n
    r_divider_copy = Signal(intbv(0, 0, 2**M))
    r_quotient = Signal(intbv(0, 0, 2**n))
    r_reminder = Signal(intbv(0, 0, 2**M))
    r_quotient_out  = Signal(intbv(0, 0, 2**n))
    r_reminder_out = Signal(intbv(0, 0, 2**n))
    tmp = int(math.log(n,2)+1)
    r_cnt = Signal(intbv(0,0,n))
    w_dif = Signal(intbv(0, -2**M, 2**M))
    
    @always_comb
    def comb():
        w_dif.next = r_reminder - (r_divider_copy >> r_cnt)
        o_quotient.next = r_quotient_out
        o_reminder.next = r_reminder_out
    @always(i_clk.posedge, i_rst.negedge)
    def seq():
        if i_rst == 0:
            r_cnt.next = 0
            r_quotient.next = 0
            r_reminder.next = 0
            r_divider_copy.next = 0
            r_quotient_out.next = 0
            r_reminder_out.next = 0
        elif i_en == 1:
            r_cnt.next = (r_cnt + 1) %(n)
            if r_cnt == 0:
                r_quotient.next = 0
                r_reminder.next = i_divident
                r_divider_copy.next = i_divider << n-1
                r_quotient_out.next = r_quotient
                r_reminder_out.next = r_reminder[n:]
            else:
                if w_dif >= 0:
                    r_quotient.next = (r_quotient << 1) + 1
                    r_reminder.next = w_dif
                else:
                    r_quotient.next = r_quotient << 1

    return comb, seq

def Cordic(i_clk, i_rst, i_en, i_arg, o_Re, o_Im, DAT_WIDTH = 16, ARG_WIDTH = 16):
    """
        i_clk - input clk
        Signal(bool(0))
        i_rst - sync reset
        Signal(bool(0))
        i_en - enable
        Signal(bool(0))
        i_arg - unsigned argument in 0..2*pi

    """
    DAT_SIGNED = DAT_WIDTH - 1
    ARG_SIGNED = ARG_WIDTH - 1

    ARG_MASK = (2**(ARG_WIDTH-2)) - 1
    QUAD_MASK = ((2**(ARG_WIDTH)) - 1) - ((2**(ARG_WIDTH-2)) - 1)

    MIN_ARG = 0
    MAX_ARG = 2**ARG_WIDTH
    
    MIN_INPUT_ARG = 0;
    MAX_INPUT_ARG = 2**(ARG_WIDTH-2)
    
    MIN_OUTPUT_ARG = -2**(ARG_WIDTH-2)
    MAX_OUTPUT_ARG = 2**(ARG_WIDTH-2)
    
    MIN_CALCULATE_ARG = -2**(ARG_WIDTH-1)
    MAX_CALCULATE_ARG = 2**(ARG_WIDTH-1)
    
    MIN_INPUT_DAT = -2**DAT_SIGNED
    MAX_INPUT_DAT = 2**DAT_SIGNED
    
    MIN_CALCULATE_DAT = -2**DAT_WIDTH
    MAX_CALCULATE_DAT = 2**DAT_WIDTH
    
    N = 24
    tmp_integer_angle  = np.arctan(1.0/np.power(2.0, np.arange(N)))*(180.0/np.pi)
    tmp = np.cos(tmp_integer_angle*np.pi/180.0)
    tmp_integer_angle *=  MAX_ARG / 360.0
    tmp_integer_angle = np.around(tmp_integer_angle, decimals = 0)
        
    int_angle = []
    for i in range(N):
        if tmp_integer_angle[i] >= 1.0:
            int_angle.append(tmp_integer_angle[i])

    N = len(int_angle)
    tmp_integer_angle  = np.arctan(1.0/np.power(2.0, np.arange(N)))*(180.0/np.pi)
    tmp = np.cos(tmp_integer_angle*np.pi/180.0)
    tmp_integer_angle *=  MAX_ARG / 360.0
    tmp_integer_angle = np.around(tmp_integer_angle, decimals = 0)

    int_angle = []
    for i in range(N):
        int_angle.append(tmp_integer_angle[i])

    print "CORDIC DEPTH: ", N
    print int_angle
    
    CORDIC_GAIN = int(tmp.prod()*(MAX_INPUT_DAT-1))
    print CORDIC_GAIN
    
    r_Re = [Signal(intbv(0, MIN_INPUT_DAT, MAX_INPUT_DAT)) for i in range(N+1)]
    w_Re = [Signal(intbv(0, MIN_CALCULATE_DAT, MAX_CALCULATE_DAT)) for i in range(N)]
    r_Im = [Signal(intbv(0, MIN_INPUT_DAT, MAX_INPUT_DAT)) for i in range(N+1)]
    w_Im = [Signal(intbv(0, MIN_CALCULATE_DAT, MAX_CALCULATE_DAT)) for i in range(N)]

    r_input_arg = [Signal(intbv(0, MIN_CALCULATE_ARG, MAX_CALCULATE_ARG)) for i in range(N+1)]
    r_output_arg = [Signal(intbv(0, MIN_CALCULATE_ARG, MAX_CALCULATE_ARG)) for i in range(N)]
    r_quad = [Signal(intbv(0, 0, 4)) for i in range(N)]
    angle = [Signal(intbv(0, MIN_CALCULATE_ARG, MAX_CALCULATE_ARG)) for i in range(N)]    
    
    ROM =  tuple([int(tmp_integer_angle[i]) for i in range(N)])
    #ROM =  tuple([(if int(tmp_integer_angle[i]) == 0: 1 else: int(tmp_integer_angle[i])) for i in range(N)])
    My_const = Signal(intbv(1, MIN_INPUT_DAT, MAX_INPUT_DAT))
    
    @always_comb
    def exits():
        o_Re.next = r_Re[N]
        o_Im.next = r_Im[N]

    @always_comb
    def comb():
        for i in range(1,N):
            w_Re[i].next = ((r_Re[i-1] + (1 << (i-1))) >> (i))
            w_Im[i].next = ((r_Im[i-1] + (1 << (i-1))) >> (i))

    @always(i_clk.posedge, i_rst.negedge)
    def seq():
        if i_rst == 0:    
            for i in range(N):
                r_input_arg[i].next = 0
                r_output_arg[i].next = 0
                r_quad[i].next = 0
                r_Re[i].next = 0
                r_Im[i].next = 0
                angle[i].next = ROM[i]
        else:
            if i_en == 1:
                r_input_arg[0].next = i_arg[ARG_WIDTH-2:]
                r_output_arg[0].next = angle[0]
                r_quad[0].next = i_arg[ARG_WIDTH:ARG_WIDTH-2]
                r_Re[0].next = CORDIC_GAIN
                r_Im[0].next = CORDIC_GAIN
                
                for i in range(1,N):
                    r_input_arg[i].next = r_input_arg[i-1]
                    r_quad[i].next = r_quad[i-1]
                    if r_output_arg[i-1] > r_input_arg[i-1]:
                        r_Re[i].next = r_Re[i-1] + w_Im[i]
                        r_Im[i].next = r_Im[i-1] - w_Re[i]
                        r_output_arg[i].next = r_output_arg[i-1] - angle[i]
                    else:
                        r_Re[i].next = r_Re[i-1] - w_Im[i]
                        r_Im[i].next = r_Im[i-1] + w_Re[i]
                        r_output_arg[i].next = r_output_arg[i-1] + angle[i]
                if r_quad[N-1] == 0:
                    r_Re[N].next = r_Re[N-1]
                    r_Im[N].next = r_Im[N-1]
                elif r_quad[N-1] == 1:
                    r_Re[N].next = -r_Im[N-1]
                    r_Im[N].next = r_Re[N-1]
                elif r_quad[N-1] == 2:
                    r_Re[N].next = -r_Re[N-1]
                    r_Im[N].next = -r_Im[N-1]
                elif r_quad[N-1] == 3:
                    r_Re[N].next = r_Im[N-1]
                    r_Im[N].next = -r_Re[N-1]
    
    return  comb, seq, exits



def Cordic_generator(i_clk, i_rst, i_en, i_freq, i_discr_freq, o_Re, o_Im, Nbits):
    ARG_IN = max(len(i_freq), len(i_discr_freq), 16)
    ARG_IN_DIV = 2*ARG_IN
    ARG_OUT = ARG_IN_DIV - ARG_IN
    
    DIVIDENT = Signal(intbv(0,0,2**ARG_IN_DIV))
    
    @always_comb
    def comb():
        DIVIDENT.next = i_freq << ARG_IN

    
    o_quotient = Signal(intbv(0,0, 2**ARG_OUT))
    o_reminder = Signal(intbv(0,0, 2**ARG_OUT))


    uut_0 = Divider(i_clk, i_rst, i_en, DIVIDENT, i_discr_freq, o_quotient, o_reminder)

    r_quo = Signal(intbv(0, 0, 2**ARG_OUT))
    w_quo_next = Signal(intbv(0, 0, 2**(ARG_OUT+1)))
    r_rem = Signal(intbv(0, 0, 2**ARG_OUT))
    w_rem_next = Signal(intbv(0, 0, 2**(ARG_OUT+1)))
    DISCR_FREQ = Signal(intbv(0,0,2**ARG_OUT))
    @always_comb
    def comb():
        DIVIDENT.next = i_freq << ARG_IN
        DISCR_FREQ.next = i_discr_freq
        w_rem_next.next = r_rem + o_reminder

    @always_comb
    def comb2():
        if w_rem_next >= DISCR_FREQ:
            w_quo_next.next = r_quo + o_quotient + 1
        else:
            w_quo_next.next = r_quo + o_quotient

    @always(i_clk.posedge)#, i_rst.negedge)
    def seq():
        if w_rem_next >= DISCR_FREQ:
            r_rem.next = w_rem_next - DISCR_FREQ
            if w_quo_next[ARG_OUT+1:ARG_OUT] == 1:
                r_quo.next = w_quo_next - 2**ARG_OUT
            else:
                r_quo.next = w_quo_next
        else:
            r_rem.next = w_rem_next
            if w_quo_next[ARG_OUT+1:ARG_OUT] == 1:
                r_quo.next = w_quo_next -  2**ARG_OUT
            else:
                r_quo.next = w_quo_next

    uut_1 = Cordic(i_clk, i_rst, i_en, r_quo, o_Re, o_Im, Nbits, ARG_IN)
    return comb, comb2, seq, uut_0, uut_1

plt_clk = []
plt_cnt = []

def tb(n = 16, f = 250, F = 8000):
    global plt_clk, plt_cnt
    #n = 16
    n_signed = n - 1
    i_clk = Signal(bool(0))
    i_rst = Signal(bool(0))
    i_en = Signal(bool(0))
    o_Re = Signal(intbv(0, -2**n_signed, 2**n_signed))
    o_Im = Signal(intbv(0, -2**n_signed, 2**n_signed))

    freq = Signal(intbv(f,0,2**f.bit_length()))
    discr_freq = Signal(intbv(F, 0, 2**F.bit_length()))
    uut = Cordic_generator(i_clk, i_rst, i_en, freq, discr_freq, o_Re, o_Im, n)
    @instance
    def control():
        while True:
            yield i_clk.posedge
            i_rst.next = 1
            yield i_clk.posedge
            yield i_clk.posedge
            yield i_clk.posedge
            i_en.next = 1

    @instance
    def clk_gen():
        while True:            
            yield delay(1)
            i_clk.next = not i_clk

    @instance
    def monitor():
        while True:
            yield i_clk.posedge
            yield delay(1)
            if i_en == 1:
#            if delay_cnt == 20+1:
                plt_clk.append(int(o_Im))
                plt_cnt.append(int(o_Re))
#                print "%d_%d__%d" %(now(), o_Re, o_Im)
    return control, clk_gen, monitor, uut

###################################
#n = 16
#n_signed = n - 1
#i_clk = Signal(bool(0))
#i_rst = Signal(bool(0))
#i_en = Signal(bool(0))
#o_Re = Signal(intbv(0, -2**n_signed, 2**n_signed))
#o_Im = Signal(intbv(0, -2**n_signed, 2**n_signed))
#f = 2000
#F = 8000
#freq = Signal(intbv(f,0,2**f.bit_length()))
#discr_freq = Signal(intbv(F, 0, 2**F.bit_length()))
#uut = toVerilog(Cordic_generator, i_clk, i_rst, i_en, freq, discr_freq, o_Re, o_Im, n)
#################################


#################################
#N = 16
#f = 500
#F = 8000
#inst = traceSignals(tb, N, f, F)
#sim = Simulation(inst)
#sim.run(10000)
#################################

#################################
#N_FFT = 128
#fig, ax = plt.subplots(3,1)
#ax[0].plot(plt_clk[-1-N_FFT:-1:1],'ro-')
#ax[1].plot(plt_cnt[-1-N_FFT:-1:1],'bo-')
#
#N_FFT = 4096
#freq = [float(i)*F/N_FFT for i in range(N_FFT)]
#X = np.fft.fft(plt_cnt[-1-N_FFT:-1:1]) + float(N_FFT)/1000
#mags = abs(X) / max(abs(X))
#X_db = 20*np.log10(mags)
#ax[2].plot(freq, X_db)
#fig.tight_layout()
#################################

#################################
#from scipy.io import wavfile
#tmp = np.array([])
#for i in range(len(plt_cnt)):
#    tmp = np.append(tmp, float(plt_cnt[i])/(2**13))
#wavfile.write('Re_tst1.wav', 48000, tmp)
#################################