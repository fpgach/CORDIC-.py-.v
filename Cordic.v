`timescale 1ns/1ps
module Cordic
#(parameter N = 14, DAT_WIDTH = 16, ARG_WIDTH = 16)
	(
	input	wire									clk,
	input	wire			[ ARG_WIDTH-1	:  0 ]	arg,
	output	wire	signed	[ DAT_WIDTH-1	:  0 ]	Re_out,
	output	wire	signed	[ DAT_WIDTH-1	:  0 ]	Im_out
	);
localparam CORDIC_GAIN = 16'd19897;
reg		signed	[ DAT_WIDTH-1	:  0 ]	Re[  0 :  N ];
reg		signed	[ DAT_WIDTH-1	:  0 ]	Im[  0 :  N ];
reg		signed	[ ARG_WIDTH-1	:  0 ]	r_input_arg[  0 :  N ];
reg		signed	[ ARG_WIDTH-1	:  0 ]	r_output_arg[  0 : N-1];
reg				[  1 :  0 ]				r_quad[  0 : N-1];
wire	signed	[ ARG_WIDTH-1	:  0 ]	angle[  0 : N-1];
assign angle[0] = 16'd4836;
assign angle[1] = 16'd2555;
assign angle[2] = 16'd1297;
assign angle[3] = 16'd651;
assign angle[4] = 16'd326;
assign angle[5] = 16'd163;
assign angle[6] = 16'd81;
assign angle[7] = 16'd41;
assign angle[8] = 16'd20;
assign angle[9] = 16'd10;
assign angle[10] = 16'd5;
assign angle[11] = 16'd3;
assign angle[12] = 16'd1;
wire    signed	[ DAT_WIDTH	:  0 ]	w_Re[0:N-1];
wire    signed	[ DAT_WIDTH	:  0 ]	w_Im[0:N-1];
genvar i;
generate
	for(i = 1; i < N; i = i + 1)
	begin: shift
		assign w_Im[i-1] = (Im[i-1] + (16'sd1 <<< (i-1))) >>> (i);
		assign w_Re[i-1] = (Re[i-1] + (16'sd1 <<< (i-1))) >>> (i);
	end
endgenerate
integer k;
always@(posedge clk)
begin
	r_input_arg[0] <= {2'b0,arg[(ARG_WIDTH-3):0]};
 	r_quad[0] <= arg[(ARG_WIDTH-1)-:2];
	Re[0] <= CORDIC_GAIN;
	Im[0] <= CORDIC_GAIN;
	r_output_arg[0] <= 16'd8192;
	for(k = 1; k < N; k = k + 1)
	begin
		r_input_arg[k] <= r_input_arg[k-1];
		r_quad[k] <= r_quad[k-1];
		if(r_output_arg[k-1] > r_input_arg[k-1])
		begin
			Re[k] <= Re[k-1] + w_Im[k-1][15:0];
			Im[k] <= Im[k-1] - w_Re[k-1][15:0];
			r_output_arg[k] <= r_output_arg[k-1] - angle[k-1];
		end
		else
		begin
			Re[k] <= Re[k-1] - w_Im[k-1][15:0];
			Im[k] <= Im[k-1] + w_Re[k-1][15:0];
			r_output_arg[k] <= r_output_arg[k-1] + angle[k-1];
		end
	end
	Re[N] <=	r_quad[N-1] == 2'b00 ? Re[N-1]	:
				r_quad[N-1] == 2'b01 ? -Im[N-1]	:
				r_quad[N-1] == 2'b10 ? -Re[N-1]	:
				Im[N-1];
	Im[N] <=	r_quad[N-1] == 2'b00 ? Im[N-1]	:
				r_quad[N-1] == 2'b01 ? Re[N-1]	:
				r_quad[N-1] == 2'b10 ? -Im[N-1]	:
				-Re[N-1];
end
assign Re_out = Re[N];
assign Im_out = Im[N];
endmodule
