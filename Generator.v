`timescale 1ns/1ps
module Generator
#(parameter CLK_RATE = 20000000, N = 14, DAT_WIDTH = 16, ARG_WIDTH = 16)
	(
	input	wire									clk,
	input	wire			[ DAT_WIDTH-1	:  0 ]	out_freq,
	input	wire			[ DAT_WIDTH-1	:  0 ]	discr_freq,
	output	wire	signed	[ DAT_WIDTH-1	:  0 ]	Re_out,
	output	wire	signed	[ DAT_WIDTH-1	:  0 ]	Im_out
	);
wire    [ ARG_WIDTH-1  :  0 ] w_quo;
wire    [ ARG_WIDTH-1  :  0 ] w_rem;
wire    [ 2*ARG_WIDTH-1:  0 ] w_div = out_freq << ARG_WIDTH;
Divider #32 u1
(
	.clk(clk),
	.divident(w_div),
	.divider({{ARG_WIDTH{1'b0}},discr_freq}),
	.quotient(w_quo),
	.reminder(w_rem)
);
//-------------//
reg     signed  [ 31 :  0 ] r_sr = 32'b0;
wire    signed  [ 31 :  0 ] inc_sr = r_sr[31] ? discr_freq :
										discr_freq-CLK_RATE;
wire    signed  [ 31 :  0 ] tic_sr = r_sr + inc_sr;
wire                        w_en = ~r_sr[31];
reg                         r_en = 1'b0;
always@(posedge clk)
begin
	r_sr <= tic_sr;
	r_en <= w_en;
end
//-------------//
reg		[ ARG_WIDTH	:  0 ] r_quo = {(ARG_WIDTH+1){1'b0}};
wire	[ ARG_WIDTH	:  0 ] w_quo_next = r_quo + w_quo;
reg		[ ARG_WIDTH	:  0 ] r_rem = {(ARG_WIDTH+1){1'b0}};
wire	[ ARG_WIDTH	:  0 ] w_rem_next = r_rem + w_rem;
always@(posedge clk)
if(r_en)
begin
	if(w_rem_next >= {1'b0, discr_freq})
	begin
		r_rem <= w_rem_next - discr_freq;
		r_quo <= w_quo_next[ARG_WIDTH] ? 1'b1 + w_quo_next - (16'd1 << ARG_WIDTH) : 1'b1 + w_quo_next;
	end
	else
	begin
		r_rem <= w_rem_next;
		r_quo <=  w_quo_next[ARG_WIDTH] ? w_quo_next - (16'd1 << ARG_WIDTH) : w_quo_next;
	end
end
Cordic #(N, DAT_WIDTH, ARG_WIDTH) u0
(
	.clk(clk),
	.arg(r_quo),
	.Re_out(Re_out),
	.Im_out(Im_out)
);
endmodule
