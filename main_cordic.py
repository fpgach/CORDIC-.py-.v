# -*- coding: utf-8 -*-
"""
Spyder Editor
Zubarev Evgenii zubarev.e.v@gmail.com
"""


import threading
import time
import serial
import numpy as np
from scipy.io import wavfile
from matplotlib import pyplot as plt

class Divider():
    def __init__(self, divident = 1, divider = 1):
        self.divident = divident
        self.divider = divider
    
    def set_val(self, divident = 1, divider = 1):
         self.divident = divident
         self.divider = divider

    def get_val(self):
        return [int(self.divident/self.divider) , int(self.divident%self.divider)]

    #max
    def int_unsigned_division(self, N):
        divider_copy = long(self.divider) << (N-1)
        #print N
        quotient = long(0)
        reminder = long(self.divident)
        for i in range(1,N):
            #divider  = divider_copy >> i
            if((reminder - (divider_copy >> i)) < 0):
                quotient = (quotient << 1) + 0
            else:
                reminder -= divider_copy >> i
                quotient = (quotient << 1) + 1
#            print bin(divider )
#            print bin(reminder)
#            print bin(quotient)
#            print "===="
        return [int(quotient), int(reminder)]

    def Divider_verilog(self, N):
        Nlog = np.around(np.log2(N), decimals = 0)
        vout = open('Divider.v', 'w')
        vout.write('`timescale 1ns/1ps\n')
        vout.write('module Divider\n')
        vout.write('#(parameter N = ' + str(int(N)) +  ')\n' + 
        '\t(\n' +
        '\tinput\twire\t\t\t\t\tclk,\n' + 
        '\tinput\twire\t[ N-1\t:  0 ]\tdivident,\n' + 
        '\tinput\twire\t[ N-1\t:  0 ]\tdivider,\n' + 
        '\toutput\twire\t[ N-1\t:  0 ]\tquotient,\n' + 
        '\toutput\twire\t[ N-1\t:  0 ]\treminder\n' +
        '\t);\n' +
        'localparam M = 2*N;\n' + 
        "reg     [ M-1   :  0 ] r_divider_copy = {M{1'b0}};\n" + 
        "reg     [ N-1   :  0 ] r_quotient = {N{1'b0}};\n" + 
        "reg     [ M-1   :  0 ] r_reminder = {M{1'b0}};\n" +
        "\nreg     [ N-1   :  0 ] r_quotient_out = {N{1'b0}};\n" + 
        "reg     [ N-1   :  0 ] r_reminder_out = {N{1'b0}};\n" + 
        "\nreg     [  " + str(int(Nlog)) + " :  0 ] r_cnt = " +str(int(Nlog+1))+"'b0;\n" + 
        "\nwire    signed  [ M-1 :  0 ] w_diff = r_reminder - (r_divider_copy >> r_cnt);\n" + 
        "assign quotient = r_quotient_out;\n" + 
        "assign reminder = r_reminder_out;\n" + 
        "\nalways@(posedge clk)\n" + 
        "begin\n" + 
        "\tr_cnt <= r_cnt + 1'b1;\n" + 
        "\tif(r_cnt == " + str(int(Nlog+1)) + "'d" + str(int(N-1)) + ")\n" + 
        "\t\tr_cnt <= " + str(int(Nlog+1)) + "'d0;\n" + 
        "\tif(r_cnt == " + str(int(Nlog+1)) + "'d0)\n" + 
        "\tbegin\n" + 
        "\t\tr_quotient <= {N{1'b0}};\n" +
        "\t\tr_reminder <= divident;\n" + 
        "\t\tr_divider_copy <= {1'b0, divider, {N-1{1'b0}}};\n" + 
        "\t\tr_quotient_out <= r_quotient;\n" + 
        "\t\tr_reminder_out <= r_reminder[N-1:0];\n" + 
        "\tend\n" +
        "\telse\n" + 
        "\tbegin\n" + 
        "\t\tif(!w_diff[M-1])\n" + 
        "\t\tbegin\n" + 
        "\t\t\tr_reminder <= w_diff;\n" + 
        "\t\t\tr_quotient <= {r_quotient[N-2:0], 1'b1};\n" + 
        "\t\tend\n" + 
        "\t\telse\n" + 
        "\t\t\tr_quotient <= {r_quotient[N-2:0], 1'b0};\n" + 
        "\tend\n" + 
        "end\n" + 
        "endmodule\n"
        
        
        
        )
        vout.close()


class Quadrature_generator():
    def __init__(self, Amp = 1.0, freq = 0.0, phi = 0.0):
        self.Amp = Amp
        self.f = freq
        self.Arg = phi
        self.Tst = 0

    def set_val(self, Amp = 1.0, freq = 1.0, F = 1.0):
        self.Amp = Amp
        self.f = freq
        self.F = F
    
    def get_val(self):
        return self.Amp*np.exp(1j*self.Arg)
    
    def next_step(self, F):
        self.Arg += 2*np.pi*self.f/F
        return self.Amp*np.exp(1j*self.Arg)
    
    def Cordic_calculate_atan_table(self, N):
        tmp = np.zeros(N)      
        tmp = np.arctan(1.0/np.power(2.0, np.arange(N)))*(180.0/np.pi)
        return tmp  #no profit for N >= 14
    def Cordic_calculate_gain(self, angles, BITS_MODULE):
        tmp = np.cos(angles*np.pi/180.0)
        #return tmp.prod()
        return int(tmp.prod()*((1<<(BITS_MODULE-1))-1))
    def Cordic_calculate_init(self, N, BITS_MODULE, BITS_PHASE):
        self.integer_angle = np.zeros(N)
        self.Re = np.zeros(N+1)
        self.Im = np.zeros(N+1)
        self.ARG = np.zeros(N)
        self.integer_angle = self.Cordic_calculate_atan_table(N) 
        self.Re[0] = self.Cordic_calculate_gain(self.integer_angle, BITS_MODULE)
        self.Im[0] = self.Re[0]
        self.integer_angle *= (1 << BITS_PHASE) / 360.0
        self.integer_angle = np.around(self.integer_angle, decimals = 0)
        #print self.integer_angle
        self.ARG[0] = self.integer_angle[0]
    def Cordic_calculate(self, angle, N, BITS_PHASE):
        input_angle = int(angle)
        input_quad = int((input_angle & (3 << (BITS_PHASE-2))) >> (BITS_PHASE-2))
        input_angle = int(input_angle & ((1 << (BITS_PHASE-2))-1))
        #print [input_angle, input_quad]
        for i in np.arange(1,N,1):
            tmp_i = 1 << (i - 1)
            if(self.ARG[i-1] > input_angle):
                self.Re[i] = self.Re[i-1] + (int(self.Im[i-1] + tmp_i) >> i)
                self.Im[i] = self.Im[i-1] - (int(self.Re[i-1] + tmp_i) >> i)
                self.ARG[i] = self.ARG[i-1] - self.integer_angle[i]
            else:
               self.Re[i] = self.Re[i-1] - (int(self.Im[i-1] + tmp_i) >> i)
               self.Im[i] = self.Im[i-1] + (int(self.Re[i-1] + tmp_i) >> i)
               self.ARG[i] = self.ARG[i-1] + self.integer_angle[i]
            #print [i, tmp_i]
            #print [self.Re[i-1], self.Re[i], int(self.Re[i-1] + tmp_i) >> i]
            #print [self.Im[i-1], self.Im[i], int(self.Im[i-1] + tmp_i) >> i]
        #print [self.Re[N-1], self.Im[N-1]]
        if(input_quad == 0):
            self.Re[N] = self.Re[N-1]
            self.Im[N] = self.Im[N-1]
        elif(input_quad == 1):
            self.Re[N] = -self.Im[N-1]
            self.Im[N] = self.Re[N-1]
        elif(input_quad == 2):
            self.Re[N] = -self.Re[N-1]
            self.Im[N] = -self.Im[N-1]
        elif(input_quad == 3):
            self.Re[N] = self.Im[N-1]
            self.Im[N] = -self.Re[N-1]
        #print [self.Re[N], self.Im[N]]
        return [self.Re[N], self.Im[N]]
        
    def Cordic_verilog(self, N, BITS_MODULE, BITS_PHASE):
        #self.Re
        vout = open('Cordic.v', 'w')
        vout.write('`timescale 1ns/1ps\n')
        vout.write('module Cordic\n')
        vout.write('#(parameter N = ' + str(int(N)) + 
        ', DAT_WIDTH = ' + str(BITS_MODULE) + 
        ', ARG_WIDTH = ' + str(BITS_PHASE) + ')\n' + '\t(\n' + 
        '\tinput\twire\t\t\t\t\t\t\t\t\tclk,\n' + 
        '\tinput\twire\t\t\t[ ARG_WIDTH-1\t:  0 ]\targ,\n' + 
        '\toutput\twire\tsigned\t[ DAT_WIDTH-1\t:  0 ]\tRe_out,\n' + 
        '\toutput\twire\tsigned\t[ DAT_WIDTH-1\t:  0 ]\tIm_out\n' + 
        '\t);\n' + 'localparam CORDIC_GAIN = ' + str(BITS_MODULE) +
        "'d" + str(int(self.Re[0])) + ';\n' + 
        'reg\t\tsigned\t[ DAT_WIDTH-1\t:  0 ]\tRe[  0 :  N ];\n' + 
        'reg\t\tsigned\t[ DAT_WIDTH-1\t:  0 ]\tIm[  0 :  N ];\n' + 
        'reg\t\tsigned\t[ ARG_WIDTH-1\t:  0 ]\tr_input_arg[  0 :  N ];\n' + 
        'reg\t\tsigned\t[ ARG_WIDTH-1\t:  0 ]\tr_output_arg[  0 : N-1];\n' + 
        'reg\t\t\t\t[  1 :  0 ]\t\t\t\tr_quad[  0 : N-1];\n' + 
        'wire\tsigned\t[ ARG_WIDTH-1\t:  0 ]\tangle[  0 : N-1];\n'
        )
        for i in range(1,len(self.integer_angle)):
            vout.write('assign angle[' +  str(i-1) + "] = " + 
            str(BITS_PHASE) + "'d" + 
            str(int(self.integer_angle[i])) + ';\n')
        vout.write(
        "wire    signed\t[ DAT_WIDTH\t:  0 ]\tw_Re[0:N-1];\n" + 
        "wire    signed\t[ DAT_WIDTH\t:  0 ]\tw_Im[0:N-1];\n" +
        "genvar i;\ngenerate\n" + 
        "\tfor(i = 1; i < N; i = i + 1)\n\tbegin: shift\n" + 
        "\t\tassign w_Im[i-1] = (Im[i-1] + (" + str(BITS_MODULE) + "'sd1 <<< (i-1))) >>> (i);\n" + 
        "\t\tassign w_Re[i-1] = (Re[i-1] + (" + str(BITS_MODULE) + "'sd1 <<< (i-1))) >>> (i);\n" + 
        "\tend\nendgenerate\n"
        
        "integer k;\n" + 
        "always@(posedge clk)\nbegin\n" + 
        "\tr_input_arg[0] <= {2'b0,arg[(ARG_WIDTH-3):0]};\n " + 
        "\tr_quad[0] <= arg[(ARG_WIDTH-1)-:2];\n" + 
        "\tRe[0] <= CORDIC_GAIN;\n" + 
        "\tIm[0] <= CORDIC_GAIN;\n" + 
        "\tr_output_arg[0] <= " + str(BITS_PHASE) + "'d" + 
        str(int(self.integer_angle[0])) + ';\n' + 
        "\tfor(k = 1; k < N; k = k + 1)\n" + "\tbegin\n" +
        "\t\tr_input_arg[k] <= r_input_arg[k-1];\n" + 
        "\t\tr_quad[k] <= r_quad[k-1];\n" + 
        "\t\tif(r_output_arg[k-1] > r_input_arg[k-1])\n" +
        "\t\tbegin\n" + 
        "\t\t\tRe[k] <= Re[k-1] + w_Im[k-1][" + str(BITS_MODULE-1) +
        ":0];\n" + 
        "\t\t\tIm[k] <= Im[k-1] - w_Re[k-1][" + str(BITS_MODULE-1) +
        ":0];\n" + 
        "\t\t\tr_output_arg[k] <= r_output_arg[k-1] - angle[k-1];\n" + 
        "\t\tend\n\t\telse\n\t\tbegin\n" + 
        "\t\t\tRe[k] <= Re[k-1] - w_Im[k-1][" + str(BITS_MODULE-1) +
        ":0];\n" +  
        "\t\t\tIm[k] <= Im[k-1] + w_Re[k-1][" + str(BITS_MODULE-1) +
        ":0];\n" + 
        "\t\t\tr_output_arg[k] <= r_output_arg[k-1] + angle[k-1];\n" + 
        "\t\tend\n\tend\n" + 
        "\tRe[N] <=\tr_quad[N-1] == 2'b00 ? Re[N-1]\t:\n" + 
        "\t\t\t\tr_quad[N-1] == 2'b01 ? -Im[N-1]\t:\n" + 
        "\t\t\t\tr_quad[N-1] == 2'b10 ? -Re[N-1]\t:\n" + 
        "\t\t\t\tIm[N-1];\n" + 
        "\tIm[N] <=\tr_quad[N-1] == 2'b00 ? Im[N-1]\t:\n" + 
        "\t\t\t\tr_quad[N-1] == 2'b01 ? Re[N-1]\t:\n" + 
        "\t\t\t\tr_quad[N-1] == 2'b10 ? -Im[N-1]\t:\n" + 
        "\t\t\t\t-Re[N-1];\n" + 
        "end\n" +
        "assign Re_out = Re[N];\n" + 
        "assign Im_out = Im[N];\n" + 
        "endmodule\n"
        
        
        )
        vout.close()
    
class Generator():
    def __init__(self, f = 1.0, F = 1.0, N = 1, BITS_MODULE = 14, BITS_PHASE = 16):
        self.f = f
        self.F = F
        self.N = N
        self.BITS_MODULE = BITS_MODULE
        self.BITS_PHASE = BITS_PHASE
        divident = int((1 << self.BITS_PHASE)*self.f)
        divider = int(self.F)

        self.div = Divider(divident, divider)
        self.cordic = Quadrature_generator(1.0, self.f)
        self.cordic.Cordic_calculate_init(self.N, self.BITS_MODULE, self.BITS_PHASE)
        [self.quo, self.rem] = self.div.int_unsigned_division(2*self.BITS_PHASE)
        print [self.quo, self.rem]
        self.new_quo = 0
        self.new_rem = 0
    def get_val(self):
        return self.cordic.Cordic_calculate(self.new_quo, self.N, self.BITS_PHASE)
    def get_next_val(self):
        
        #print [self.new_quo]
        self.new_quo += self.quo
        self.new_rem += self.rem
        if(self.new_rem >= self.F):
            if(self.new_quo >= (1 << self.BITS_PHASE)):
                self.new_quo += 1 - (1 << self.BITS_PHASE)
            else:
                self.new_quo += 1
            self.new_rem -= self.F
        else:
            if(self.new_quo >= (1 << self.BITS_PHASE)):
                self.new_quo -= 1 << self.BITS_PHASE
        #print self.cordic.Cordic_calculate(self.new_quo, self.N, self.BITS_PHASE)
        return self.cordic.Cordic_calculate(self.new_quo, self.N, self.BITS_PHASE)

    def Generator_verilog(self, clk):
        self.cordic.Cordic_verilog(self.N, self.BITS_MODULE, self.BITS_PHASE)
        self.div.Divider_verilog(2*self.BITS_MODULE)
        vout = open('Generator.v', 'w')
        vout.write('`timescale 1ns/1ps\n')
        vout.write('module Generator\n')
        vout.write('#(parameter CLK_RATE = '+ str(int(clk))  + 
        ', N = ' + str(int(self.N)) + 
        ', DAT_WIDTH = ' + str(self.BITS_MODULE) + 
        ', ARG_WIDTH = ' + str(self.BITS_PHASE) + ')\n' + 
        "\t(\n" + 
        '\tinput\twire\t\t\t\t\t\t\t\t\tclk,\n' + 
        '\tinput\twire\t\t\t[ DAT_WIDTH-1\t:  0 ]\tout_freq,\n' + 
        '\tinput\twire\t\t\t[ DAT_WIDTH-1\t:  0 ]\tdiscr_freq,\n' + 
        '\toutput\twire\tsigned\t[ DAT_WIDTH-1\t:  0 ]\tRe_out,\n' + 
        '\toutput\twire\tsigned\t[ DAT_WIDTH-1\t:  0 ]\tIm_out\n' +
        "\t);\n" + 
        "wire    [ ARG_WIDTH-1  :  0 ] w_quo;\n" + 
        "wire    [ ARG_WIDTH-1  :  0 ] w_rem;\n" + 
        "wire    [ 2*ARG_WIDTH-1:  0 ] w_div = out_freq << ARG_WIDTH;\n"
        "Divider #" + str(int(2*self.BITS_PHASE)) + " u1\n" + 
        "(\n" + "\t.clk(clk),\n" + 
        "\t.divident(w_div),\n" + 
        "\t.divider({{ARG_WIDTH{1'b0}},discr_freq}),\n" + 
        "\t.quotient(w_quo),\n" + 
        "\t.reminder(w_rem)\n);\n" + 
        "//-------------//\nreg     signed  [ 31 :  0 ] r_sr = 32'b0;\n" + 
        "wire    signed  [ 31 :  0 ] inc_sr = r_sr[31] ? discr_freq :\n" +
        "\t\t\t\t\t\t\t\t\t\tdiscr_freq-CLK_RATE;\n" +
        "wire    signed  [ 31 :  0 ] tic_sr = r_sr + inc_sr;\n" + 
        "wire                        w_en = ~r_sr[31];\n" + 
        "reg                         r_en = 1'b0;\n" + 
        "always@(posedge clk)\n" + 
        "begin\n" + 
        "\tr_sr <= tic_sr;\n" + 
        "\tr_en <= w_en;\n" +
        "end\n//-------------//\n" + 
        "reg\t\t[ ARG_WIDTH\t:  0 ] r_quo = {(ARG_WIDTH+1){1'b0}};\n" + 
        "wire\t[ ARG_WIDTH\t:  0 ] w_quo_next = r_quo + w_quo;\n" + 
        "reg\t\t[ ARG_WIDTH\t:  0 ] r_rem = {(ARG_WIDTH+1){1'b0}};\n" + 
        "wire\t[ ARG_WIDTH\t:  0 ] w_rem_next = r_rem + w_rem;\n" + 
        
        
        "always@(posedge clk)\n" + 
        "if(r_en)\n" + 
        "begin\n" +
        "\tif(w_rem_next >= {1'b0, discr_freq})\n" + 
        "\tbegin\n" + 
        "\t\tr_rem <= w_rem_next - discr_freq;\n" + 
        "\t\tr_quo <= w_quo_next[ARG_WIDTH] ? 1'b1 + w_quo_next - (" + 
        str(self.BITS_PHASE) + "'d1 << ARG_WIDTH) : 1'b1 + w_quo_next;\n" + 
        "\tend\n" + 
        "\telse\n\tbegin\n" + 
        "\t\tr_rem <= w_rem_next;\n" + 
        "\t\tr_quo <=  w_quo_next[ARG_WIDTH] ? w_quo_next - (" + 
        str(self.BITS_PHASE) + "'d1 << ARG_WIDTH) : w_quo_next;\n" + 
        "\tend\n"+
        "end\n" +
        "Cordic #(N, DAT_WIDTH, ARG_WIDTH) u0\n(\n" + 
        "\t.clk(clk),\n" + 
        "\t.arg(r_quo),\n" + 
        "\t.Re_out(Re_out),\n" +
        "\t.Im_out(Im_out)\n" + 
        ");\n" + 
        "endmodule\n"
        )
        vout.close()

        

    def get_values(self, N):
        tmp = [0]*N
        result = self.get_val()
        tmp[0] = complex(result[0], result[1])
        for i in range(1,N,1):
            result = self.get_next_val()
            tmp[i] =  complex(result[0], result[1])
            #print result
        return tmp


N = 4000
n = np.arange(N)

f = 40.0
F = 8000.0
gen = Generator(f, F, 14,16,16)
gen.Generator_verilog(20000000)

result = gen.get_values(N)
#print result

X = np.array([])

X = np.fft.fft(result) + N/1000.0    #-150 dB
freq = n*F/N

#print X

mags = abs(X) / max(abs(X))
X_db = 20*np.log10(mags)
#print X_db

fig, ax = plt.subplots(1,1)
ax.plot(freq,X_db,'ro--')
