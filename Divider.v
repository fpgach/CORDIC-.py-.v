`timescale 1ns/1ps
module Divider
#(parameter N = 32)
	(
	input	wire					clk,
	input	wire	[ N-1	:  0 ]	divident,
	input	wire	[ N-1	:  0 ]	divider,
	output	wire	[ N-1	:  0 ]	quotient,
	output	wire	[ N-1	:  0 ]	reminder
	);
localparam M = 2*N;
reg     [ M-1   :  0 ] r_divider_copy = {M{1'b0}};
reg     [ N-1   :  0 ] r_quotient = {N{1'b0}};
reg     [ M-1   :  0 ] r_reminder = {M{1'b0}};

reg     [ N-1   :  0 ] r_quotient_out = {N{1'b0}};
reg     [ N-1   :  0 ] r_reminder_out = {N{1'b0}};

reg     [  5 :  0 ] r_cnt = 6'b0;

wire    signed  [ M-1 :  0 ] w_diff = r_reminder - (r_divider_copy >> r_cnt);
assign quotient = r_quotient_out;
assign reminder = r_reminder_out;

always@(posedge clk)
begin
	r_cnt <= r_cnt + 1'b1;
	if(r_cnt == 6'd31)
		r_cnt <= 6'd0;
	if(r_cnt == 6'd0)
	begin
		r_quotient <= {N{1'b0}};
		r_reminder <= divident;
		r_divider_copy <= {1'b0, divider, {N-1{1'b0}}};
		r_quotient_out <= r_quotient;
		r_reminder_out <= r_reminder[N-1:0];
	end
	else
	begin
		if(!w_diff[M-1])
		begin
			r_reminder <= w_diff;
			r_quotient <= {r_quotient[N-2:0], 1'b1};
		end
		else
			r_quotient <= {r_quotient[N-2:0], 1'b0};
	end
end
endmodule
