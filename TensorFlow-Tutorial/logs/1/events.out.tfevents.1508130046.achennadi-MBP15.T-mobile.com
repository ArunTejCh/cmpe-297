       �K"	  �?y�Abrain.Event:2����=      �3�J	}��?y�A"�{
i
inputsPlaceholder*
dtype0*
shape:���������
*'
_output_shapes
:���������

a
outputPlaceholder*
dtype0*
shape:���������*#
_output_shapes
:���������
U
one_hot/on_valueConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
V
one_hot/off_valueConst*
valueB
 *    *
dtype0*
_output_shapes
: 
O
one_hot/depthConst*
value	B :*
dtype0*
_output_shapes
: 
�
one_hotOneHotoutputone_hot/depthone_hot/on_valueone_hot/off_value*
axis���������*
T0*
TI0*'
_output_shapes
:���������
g
truncated_normal/shapeConst*
valueB"
      *
dtype0*
_output_shapes
:
Z
truncated_normal/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
\
truncated_normal/stddevConst*
valueB
 *���=*
dtype0*
_output_shapes
: 
�
 truncated_normal/TruncatedNormalTruncatedNormaltruncated_normal/shape*

seed *
seed2 *
dtype0*
T0*
_output_shapes

:


truncated_normal/mulMul truncated_normal/TruncatedNormaltruncated_normal/stddev*
T0*
_output_shapes

:

m
truncated_normalAddtruncated_normal/multruncated_normal/mean*
T0*
_output_shapes

:

|
Variable
VariableV2*
shape
:
*
dtype0*
	container *
shared_name *
_output_shapes

:

�
Variable/AssignAssignVariabletruncated_normal*
T0*
validate_shape(*
use_locking(*
_class
loc:@Variable*
_output_shapes

:

i
Variable/readIdentityVariable*
T0*
_class
loc:@Variable*
_output_shapes

:

i
truncated_normal_1/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
\
truncated_normal_1/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
^
truncated_normal_1/stddevConst*
valueB
 *���=*
dtype0*
_output_shapes
: 
�
"truncated_normal_1/TruncatedNormalTruncatedNormaltruncated_normal_1/shape*

seed *
seed2 *
dtype0*
T0*
_output_shapes

:
�
truncated_normal_1/mulMul"truncated_normal_1/TruncatedNormaltruncated_normal_1/stddev*
T0*
_output_shapes

:
s
truncated_normal_1Addtruncated_normal_1/multruncated_normal_1/mean*
T0*
_output_shapes

:
~

Variable_1
VariableV2*
shape
:*
dtype0*
	container *
shared_name *
_output_shapes

:
�
Variable_1/AssignAssign
Variable_1truncated_normal_1*
T0*
validate_shape(*
use_locking(*
_class
loc:@Variable_1*
_output_shapes

:
o
Variable_1/readIdentity
Variable_1*
T0*
_class
loc:@Variable_1*
_output_shapes

:
R
zerosConst*
valueB*    *
dtype0*
_output_shapes
:
v

Variable_2
VariableV2*
shape:*
dtype0*
	container *
shared_name *
_output_shapes
:
�
Variable_2/AssignAssign
Variable_2zeros*
T0*
validate_shape(*
use_locking(*
_class
loc:@Variable_2*
_output_shapes
:
k
Variable_2/readIdentity
Variable_2*
T0*
_class
loc:@Variable_2*
_output_shapes
:
T
zeros_1Const*
valueB*    *
dtype0*
_output_shapes
:
v

Variable_3
VariableV2*
shape:*
dtype0*
	container *
shared_name *
_output_shapes
:
�
Variable_3/AssignAssign
Variable_3zeros_1*
T0*
validate_shape(*
use_locking(*
_class
loc:@Variable_3*
_output_shapes
:
k
Variable_3/readIdentity
Variable_3*
T0*
_class
loc:@Variable_3*
_output_shapes
:

MatMulMatMulinputsVariable/read*
transpose_a( *
transpose_b( *
T0*'
_output_shapes
:���������
t
BiasAddBiasAddMatMulVariable_2/read*
T0*
data_formatNHWC*'
_output_shapes
:���������
G
ReluReluBiasAdd*
T0*'
_output_shapes
:���������
�
MatMul_1MatMulReluVariable_1/read*
transpose_a( *
transpose_b( *
T0*'
_output_shapes
:���������
x
	BiasAdd_1BiasAddMatMul_1Variable_3/read*
T0*
data_formatNHWC*'
_output_shapes
:���������
F
RankConst*
value	B :*
dtype0*
_output_shapes
: 
N
ShapeShape	BiasAdd_1*
T0*
out_type0*
_output_shapes
:
H
Rank_1Const*
value	B :*
dtype0*
_output_shapes
: 
P
Shape_1Shape	BiasAdd_1*
T0*
out_type0*
_output_shapes
:
G
Sub/yConst*
value	B :*
dtype0*
_output_shapes
: 
:
SubSubRank_1Sub/y*
T0*
_output_shapes
: 
R
Slice/beginPackSub*
N*
T0*

axis *
_output_shapes
:
T

Slice/sizeConst*
valueB:*
dtype0*
_output_shapes
:
b
SliceSliceShape_1Slice/begin
Slice/size*
T0*
Index0*
_output_shapes
:
b
concat/values_0Const*
valueB:
���������*
dtype0*
_output_shapes
:
M
concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
q
concatConcatV2concat/values_0Sliceconcat/axis*
N*
T0*

Tidx0*
_output_shapes
:
n
ReshapeReshape	BiasAdd_1concat*
T0*
Tshape0*0
_output_shapes
:������������������
H
Rank_2Const*
value	B :*
dtype0*
_output_shapes
: 
N
Shape_2Shapeone_hot*
T0*
out_type0*
_output_shapes
:
I
Sub_1/yConst*
value	B :*
dtype0*
_output_shapes
: 
>
Sub_1SubRank_2Sub_1/y*
T0*
_output_shapes
: 
V
Slice_1/beginPackSub_1*
N*
T0*

axis *
_output_shapes
:
V
Slice_1/sizeConst*
valueB:*
dtype0*
_output_shapes
:
h
Slice_1SliceShape_2Slice_1/beginSlice_1/size*
T0*
Index0*
_output_shapes
:
d
concat_1/values_0Const*
valueB:
���������*
dtype0*
_output_shapes
:
O
concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
y
concat_1ConcatV2concat_1/values_0Slice_1concat_1/axis*
N*
T0*

Tidx0*
_output_shapes
:
p
	Reshape_1Reshapeone_hotconcat_1*
T0*
Tshape0*0
_output_shapes
:������������������
�
SoftmaxCrossEntropyWithLogitsSoftmaxCrossEntropyWithLogitsReshape	Reshape_1*
T0*?
_output_shapes-
+:���������:������������������
I
Sub_2/yConst*
value	B :*
dtype0*
_output_shapes
: 
<
Sub_2SubRankSub_2/y*
T0*
_output_shapes
: 
W
Slice_2/beginConst*
valueB: *
dtype0*
_output_shapes
:
U
Slice_2/sizePackSub_2*
N*
T0*

axis *
_output_shapes
:
o
Slice_2SliceShapeSlice_2/beginSlice_2/size*
T0*
Index0*#
_output_shapes
:���������
x
	Reshape_2ReshapeSoftmaxCrossEntropyWithLogitsSlice_2*
T0*
Tshape0*#
_output_shapes
:���������
O
ConstConst*
valueB: *
dtype0*
_output_shapes
:
\
MeanMean	Reshape_2Const*
	keep_dims( *
T0*

Tidx0*
_output_shapes
: 
R
gradients/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
T
gradients/ConstConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
Y
gradients/FillFillgradients/Shapegradients/Const*
T0*
_output_shapes
: 
k
!gradients/Mean_grad/Reshape/shapeConst*
valueB:*
dtype0*
_output_shapes
:
�
gradients/Mean_grad/ReshapeReshapegradients/Fill!gradients/Mean_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
:
b
gradients/Mean_grad/ShapeShape	Reshape_2*
T0*
out_type0*
_output_shapes
:
�
gradients/Mean_grad/TileTilegradients/Mean_grad/Reshapegradients/Mean_grad/Shape*
T0*

Tmultiples0*#
_output_shapes
:���������
d
gradients/Mean_grad/Shape_1Shape	Reshape_2*
T0*
out_type0*
_output_shapes
:
^
gradients/Mean_grad/Shape_2Const*
valueB *
dtype0*
_output_shapes
: 
c
gradients/Mean_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:
�
gradients/Mean_grad/ProdProdgradients/Mean_grad/Shape_1gradients/Mean_grad/Const*
	keep_dims( *
T0*

Tidx0*
_output_shapes
: 
e
gradients/Mean_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
�
gradients/Mean_grad/Prod_1Prodgradients/Mean_grad/Shape_2gradients/Mean_grad/Const_1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
: 
_
gradients/Mean_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 
�
gradients/Mean_grad/MaximumMaximumgradients/Mean_grad/Prod_1gradients/Mean_grad/Maximum/y*
T0*
_output_shapes
: 
�
gradients/Mean_grad/floordivFloorDivgradients/Mean_grad/Prodgradients/Mean_grad/Maximum*
T0*
_output_shapes
: 
n
gradients/Mean_grad/CastCastgradients/Mean_grad/floordiv*

SrcT0*

DstT0*
_output_shapes
: 
�
gradients/Mean_grad/truedivRealDivgradients/Mean_grad/Tilegradients/Mean_grad/Cast*
T0*#
_output_shapes
:���������
{
gradients/Reshape_2_grad/ShapeShapeSoftmaxCrossEntropyWithLogits*
T0*
out_type0*
_output_shapes
:
�
 gradients/Reshape_2_grad/ReshapeReshapegradients/Mean_grad/truedivgradients/Reshape_2_grad/Shape*
T0*
Tshape0*#
_output_shapes
:���������
}
gradients/zeros_like	ZerosLikeSoftmaxCrossEntropyWithLogits:1*
T0*0
_output_shapes
:������������������
�
;gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDims/dimConst*
valueB :
���������*
dtype0*
_output_shapes
: 
�
7gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDims
ExpandDims gradients/Reshape_2_grad/Reshape;gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDims/dim*
T0*

Tdim0*'
_output_shapes
:���������
�
0gradients/SoftmaxCrossEntropyWithLogits_grad/mulMul7gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDimsSoftmaxCrossEntropyWithLogits:1*
T0*0
_output_shapes
:������������������
e
gradients/Reshape_grad/ShapeShape	BiasAdd_1*
T0*
out_type0*
_output_shapes
:
�
gradients/Reshape_grad/ReshapeReshape0gradients/SoftmaxCrossEntropyWithLogits_grad/mulgradients/Reshape_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
$gradients/BiasAdd_1_grad/BiasAddGradBiasAddGradgradients/Reshape_grad/Reshape*
T0*
data_formatNHWC*
_output_shapes
:
y
)gradients/BiasAdd_1_grad/tuple/group_depsNoOp^gradients/Reshape_grad/Reshape%^gradients/BiasAdd_1_grad/BiasAddGrad
�
1gradients/BiasAdd_1_grad/tuple/control_dependencyIdentitygradients/Reshape_grad/Reshape*^gradients/BiasAdd_1_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/Reshape_grad/Reshape*'
_output_shapes
:���������
�
3gradients/BiasAdd_1_grad/tuple/control_dependency_1Identity$gradients/BiasAdd_1_grad/BiasAddGrad*^gradients/BiasAdd_1_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients/BiasAdd_1_grad/BiasAddGrad*
_output_shapes
:
�
gradients/MatMul_1_grad/MatMulMatMul1gradients/BiasAdd_1_grad/tuple/control_dependencyVariable_1/read*
transpose_a( *
transpose_b(*
T0*'
_output_shapes
:���������
�
 gradients/MatMul_1_grad/MatMul_1MatMulRelu1gradients/BiasAdd_1_grad/tuple/control_dependency*
transpose_a(*
transpose_b( *
T0*
_output_shapes

:
t
(gradients/MatMul_1_grad/tuple/group_depsNoOp^gradients/MatMul_1_grad/MatMul!^gradients/MatMul_1_grad/MatMul_1
�
0gradients/MatMul_1_grad/tuple/control_dependencyIdentitygradients/MatMul_1_grad/MatMul)^gradients/MatMul_1_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/MatMul_1_grad/MatMul*'
_output_shapes
:���������
�
2gradients/MatMul_1_grad/tuple/control_dependency_1Identity gradients/MatMul_1_grad/MatMul_1)^gradients/MatMul_1_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients/MatMul_1_grad/MatMul_1*
_output_shapes

:
�
gradients/Relu_grad/ReluGradReluGrad0gradients/MatMul_1_grad/tuple/control_dependencyRelu*
T0*'
_output_shapes
:���������
�
"gradients/BiasAdd_grad/BiasAddGradBiasAddGradgradients/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
:
s
'gradients/BiasAdd_grad/tuple/group_depsNoOp^gradients/Relu_grad/ReluGrad#^gradients/BiasAdd_grad/BiasAddGrad
�
/gradients/BiasAdd_grad/tuple/control_dependencyIdentitygradients/Relu_grad/ReluGrad(^gradients/BiasAdd_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/Relu_grad/ReluGrad*'
_output_shapes
:���������
�
1gradients/BiasAdd_grad/tuple/control_dependency_1Identity"gradients/BiasAdd_grad/BiasAddGrad(^gradients/BiasAdd_grad/tuple/group_deps*
T0*5
_class+
)'loc:@gradients/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
�
gradients/MatMul_grad/MatMulMatMul/gradients/BiasAdd_grad/tuple/control_dependencyVariable/read*
transpose_a( *
transpose_b(*
T0*'
_output_shapes
:���������

�
gradients/MatMul_grad/MatMul_1MatMulinputs/gradients/BiasAdd_grad/tuple/control_dependency*
transpose_a(*
transpose_b( *
T0*
_output_shapes

:

n
&gradients/MatMul_grad/tuple/group_depsNoOp^gradients/MatMul_grad/MatMul^gradients/MatMul_grad/MatMul_1
�
.gradients/MatMul_grad/tuple/control_dependencyIdentitygradients/MatMul_grad/MatMul'^gradients/MatMul_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/MatMul_grad/MatMul*'
_output_shapes
:���������

�
0gradients/MatMul_grad/tuple/control_dependency_1Identitygradients/MatMul_grad/MatMul_1'^gradients/MatMul_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/MatMul_grad/MatMul_1*
_output_shapes

:

b
GradientDescent/learning_rateConst*
valueB
 *���=*
dtype0*
_output_shapes
: 
�
4GradientDescent/update_Variable/ApplyGradientDescentApplyGradientDescentVariableGradientDescent/learning_rate0gradients/MatMul_grad/tuple/control_dependency_1*
T0*
use_locking( *
_class
loc:@Variable*
_output_shapes

:

�
6GradientDescent/update_Variable_1/ApplyGradientDescentApplyGradientDescent
Variable_1GradientDescent/learning_rate2gradients/MatMul_1_grad/tuple/control_dependency_1*
T0*
use_locking( *
_class
loc:@Variable_1*
_output_shapes

:
�
6GradientDescent/update_Variable_2/ApplyGradientDescentApplyGradientDescent
Variable_2GradientDescent/learning_rate1gradients/BiasAdd_grad/tuple/control_dependency_1*
T0*
use_locking( *
_class
loc:@Variable_2*
_output_shapes
:
�
6GradientDescent/update_Variable_3/ApplyGradientDescentApplyGradientDescent
Variable_3GradientDescent/learning_rate3gradients/BiasAdd_1_grad/tuple/control_dependency_1*
T0*
use_locking( *
_class
loc:@Variable_3*
_output_shapes
:
�
GradientDescentNoOp5^GradientDescent/update_Variable/ApplyGradientDescent7^GradientDescent/update_Variable_1/ApplyGradientDescent7^GradientDescent/update_Variable_2/ApplyGradientDescent7^GradientDescent/update_Variable_3/ApplyGradientDescent
Z
initNoOp^Variable/Assign^Variable_1/Assign^Variable_2/Assign^Variable_3/Assign"3�n	K      ��	i�?y�AJ��
��
9
Add
x"T
y"T
z"T"
Ttype:
2	
�
ApplyGradientDescent
var"T�

alpha"T

delta"T
out"T�"
Ttype:
2	"
use_lockingbool( 
x
Assign
ref"T�

value"T

output_ref"T�"	
Ttype"
validate_shapebool("
use_lockingbool(�
{
BiasAdd

value"T	
bias"T
output"T"
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
{
BiasAddGrad
out_backprop"T
output"T"
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
4
Fill
dims

value"T
output"T"	
Ttype
>
FloorDiv
x"T
y"T
z"T"
Ttype:
2	
.
Identity

input"T
output"T"	
Ttype
o
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2
:
Maximum
x"T
y"T
z"T"
Ttype:	
2	�
�
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( "
Ttype:
2	"
Tidxtype0:
2	
<
Mul
x"T
y"T
z"T"
Ttype:
2	�

NoOp
�
OneHot
indices"TI	
depth
on_value"T
	off_value"T
output"T"
axisint���������"	
Ttype"
TItype0	:
2	
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
�
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( "
Ttype:
2	"
Tidxtype0:
2	
=
RealDiv
x"T
y"T
z"T"
Ttype:
2	
A
Relu
features"T
activations"T"
Ttype:
2		
S
ReluGrad
	gradients"T
features"T
	backprops"T"
Ttype:
2		
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
a
Slice

input"T
begin"Index
size"Index
output"T"	
Ttype"
Indextype:
2	
i
SoftmaxCrossEntropyWithLogits
features"T
labels"T	
loss"T
backprop"T"
Ttype:
2
5
Sub
x"T
y"T
z"T"
Ttype:
	2	
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	

TruncatedNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	�
s

VariableV2
ref"dtype�"
shapeshape"
dtypetype"
	containerstring "
shared_namestring �
&
	ZerosLike
x"T
y"T"	
Ttype*1.3.02v1.3.0-rc2-20-g0787eee�{
i
inputsPlaceholder*
dtype0*
shape:���������
*'
_output_shapes
:���������

a
outputPlaceholder*
dtype0*
shape:���������*#
_output_shapes
:���������
U
one_hot/on_valueConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
V
one_hot/off_valueConst*
valueB
 *    *
dtype0*
_output_shapes
: 
O
one_hot/depthConst*
value	B :*
dtype0*
_output_shapes
: 
�
one_hotOneHotoutputone_hot/depthone_hot/on_valueone_hot/off_value*
axis���������*
T0*
TI0*'
_output_shapes
:���������
g
truncated_normal/shapeConst*
valueB"
      *
dtype0*
_output_shapes
:
Z
truncated_normal/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
\
truncated_normal/stddevConst*
valueB
 *���=*
dtype0*
_output_shapes
: 
�
 truncated_normal/TruncatedNormalTruncatedNormaltruncated_normal/shape*

seed *
seed2 *
dtype0*
T0*
_output_shapes

:


truncated_normal/mulMul truncated_normal/TruncatedNormaltruncated_normal/stddev*
T0*
_output_shapes

:

m
truncated_normalAddtruncated_normal/multruncated_normal/mean*
T0*
_output_shapes

:

|
Variable
VariableV2*
shape
:
*
dtype0*
	container *
shared_name *
_output_shapes

:

�
Variable/AssignAssignVariabletruncated_normal*
T0*
validate_shape(*
use_locking(*
_class
loc:@Variable*
_output_shapes

:

i
Variable/readIdentityVariable*
T0*
_class
loc:@Variable*
_output_shapes

:

i
truncated_normal_1/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
\
truncated_normal_1/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
^
truncated_normal_1/stddevConst*
valueB
 *���=*
dtype0*
_output_shapes
: 
�
"truncated_normal_1/TruncatedNormalTruncatedNormaltruncated_normal_1/shape*

seed *
seed2 *
dtype0*
T0*
_output_shapes

:
�
truncated_normal_1/mulMul"truncated_normal_1/TruncatedNormaltruncated_normal_1/stddev*
T0*
_output_shapes

:
s
truncated_normal_1Addtruncated_normal_1/multruncated_normal_1/mean*
T0*
_output_shapes

:
~

Variable_1
VariableV2*
shape
:*
dtype0*
	container *
shared_name *
_output_shapes

:
�
Variable_1/AssignAssign
Variable_1truncated_normal_1*
T0*
validate_shape(*
use_locking(*
_class
loc:@Variable_1*
_output_shapes

:
o
Variable_1/readIdentity
Variable_1*
T0*
_class
loc:@Variable_1*
_output_shapes

:
R
zerosConst*
valueB*    *
dtype0*
_output_shapes
:
v

Variable_2
VariableV2*
shape:*
dtype0*
	container *
shared_name *
_output_shapes
:
�
Variable_2/AssignAssign
Variable_2zeros*
T0*
validate_shape(*
use_locking(*
_class
loc:@Variable_2*
_output_shapes
:
k
Variable_2/readIdentity
Variable_2*
T0*
_class
loc:@Variable_2*
_output_shapes
:
T
zeros_1Const*
valueB*    *
dtype0*
_output_shapes
:
v

Variable_3
VariableV2*
shape:*
dtype0*
	container *
shared_name *
_output_shapes
:
�
Variable_3/AssignAssign
Variable_3zeros_1*
T0*
validate_shape(*
use_locking(*
_class
loc:@Variable_3*
_output_shapes
:
k
Variable_3/readIdentity
Variable_3*
T0*
_class
loc:@Variable_3*
_output_shapes
:

MatMulMatMulinputsVariable/read*
transpose_a( *
transpose_b( *
T0*'
_output_shapes
:���������
t
BiasAddBiasAddMatMulVariable_2/read*
T0*
data_formatNHWC*'
_output_shapes
:���������
G
ReluReluBiasAdd*
T0*'
_output_shapes
:���������
�
MatMul_1MatMulReluVariable_1/read*
transpose_a( *
transpose_b( *
T0*'
_output_shapes
:���������
x
	BiasAdd_1BiasAddMatMul_1Variable_3/read*
T0*
data_formatNHWC*'
_output_shapes
:���������
F
RankConst*
value	B :*
dtype0*
_output_shapes
: 
N
ShapeShape	BiasAdd_1*
T0*
out_type0*
_output_shapes
:
H
Rank_1Const*
value	B :*
dtype0*
_output_shapes
: 
P
Shape_1Shape	BiasAdd_1*
T0*
out_type0*
_output_shapes
:
G
Sub/yConst*
value	B :*
dtype0*
_output_shapes
: 
:
SubSubRank_1Sub/y*
T0*
_output_shapes
: 
R
Slice/beginPackSub*
N*
T0*

axis *
_output_shapes
:
T

Slice/sizeConst*
valueB:*
dtype0*
_output_shapes
:
b
SliceSliceShape_1Slice/begin
Slice/size*
T0*
Index0*
_output_shapes
:
b
concat/values_0Const*
valueB:
���������*
dtype0*
_output_shapes
:
M
concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
q
concatConcatV2concat/values_0Sliceconcat/axis*
N*
T0*

Tidx0*
_output_shapes
:
n
ReshapeReshape	BiasAdd_1concat*
T0*
Tshape0*0
_output_shapes
:������������������
H
Rank_2Const*
value	B :*
dtype0*
_output_shapes
: 
N
Shape_2Shapeone_hot*
T0*
out_type0*
_output_shapes
:
I
Sub_1/yConst*
value	B :*
dtype0*
_output_shapes
: 
>
Sub_1SubRank_2Sub_1/y*
T0*
_output_shapes
: 
V
Slice_1/beginPackSub_1*
N*
T0*

axis *
_output_shapes
:
V
Slice_1/sizeConst*
valueB:*
dtype0*
_output_shapes
:
h
Slice_1SliceShape_2Slice_1/beginSlice_1/size*
T0*
Index0*
_output_shapes
:
d
concat_1/values_0Const*
valueB:
���������*
dtype0*
_output_shapes
:
O
concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
y
concat_1ConcatV2concat_1/values_0Slice_1concat_1/axis*
N*
T0*

Tidx0*
_output_shapes
:
p
	Reshape_1Reshapeone_hotconcat_1*
T0*
Tshape0*0
_output_shapes
:������������������
�
SoftmaxCrossEntropyWithLogitsSoftmaxCrossEntropyWithLogitsReshape	Reshape_1*
T0*?
_output_shapes-
+:���������:������������������
I
Sub_2/yConst*
value	B :*
dtype0*
_output_shapes
: 
<
Sub_2SubRankSub_2/y*
T0*
_output_shapes
: 
W
Slice_2/beginConst*
valueB: *
dtype0*
_output_shapes
:
U
Slice_2/sizePackSub_2*
N*
T0*

axis *
_output_shapes
:
o
Slice_2SliceShapeSlice_2/beginSlice_2/size*
T0*
Index0*#
_output_shapes
:���������
x
	Reshape_2ReshapeSoftmaxCrossEntropyWithLogitsSlice_2*
T0*
Tshape0*#
_output_shapes
:���������
O
ConstConst*
valueB: *
dtype0*
_output_shapes
:
\
MeanMean	Reshape_2Const*
	keep_dims( *
T0*

Tidx0*
_output_shapes
: 
R
gradients/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
T
gradients/ConstConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
Y
gradients/FillFillgradients/Shapegradients/Const*
T0*
_output_shapes
: 
k
!gradients/Mean_grad/Reshape/shapeConst*
valueB:*
dtype0*
_output_shapes
:
�
gradients/Mean_grad/ReshapeReshapegradients/Fill!gradients/Mean_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
:
b
gradients/Mean_grad/ShapeShape	Reshape_2*
T0*
out_type0*
_output_shapes
:
�
gradients/Mean_grad/TileTilegradients/Mean_grad/Reshapegradients/Mean_grad/Shape*
T0*

Tmultiples0*#
_output_shapes
:���������
d
gradients/Mean_grad/Shape_1Shape	Reshape_2*
T0*
out_type0*
_output_shapes
:
^
gradients/Mean_grad/Shape_2Const*
valueB *
dtype0*
_output_shapes
: 
c
gradients/Mean_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:
�
gradients/Mean_grad/ProdProdgradients/Mean_grad/Shape_1gradients/Mean_grad/Const*
	keep_dims( *
T0*

Tidx0*
_output_shapes
: 
e
gradients/Mean_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
�
gradients/Mean_grad/Prod_1Prodgradients/Mean_grad/Shape_2gradients/Mean_grad/Const_1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
: 
_
gradients/Mean_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 
�
gradients/Mean_grad/MaximumMaximumgradients/Mean_grad/Prod_1gradients/Mean_grad/Maximum/y*
T0*
_output_shapes
: 
�
gradients/Mean_grad/floordivFloorDivgradients/Mean_grad/Prodgradients/Mean_grad/Maximum*
T0*
_output_shapes
: 
n
gradients/Mean_grad/CastCastgradients/Mean_grad/floordiv*

SrcT0*

DstT0*
_output_shapes
: 
�
gradients/Mean_grad/truedivRealDivgradients/Mean_grad/Tilegradients/Mean_grad/Cast*
T0*#
_output_shapes
:���������
{
gradients/Reshape_2_grad/ShapeShapeSoftmaxCrossEntropyWithLogits*
T0*
out_type0*
_output_shapes
:
�
 gradients/Reshape_2_grad/ReshapeReshapegradients/Mean_grad/truedivgradients/Reshape_2_grad/Shape*
T0*
Tshape0*#
_output_shapes
:���������
}
gradients/zeros_like	ZerosLikeSoftmaxCrossEntropyWithLogits:1*
T0*0
_output_shapes
:������������������
�
;gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDims/dimConst*
valueB :
���������*
dtype0*
_output_shapes
: 
�
7gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDims
ExpandDims gradients/Reshape_2_grad/Reshape;gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDims/dim*
T0*

Tdim0*'
_output_shapes
:���������
�
0gradients/SoftmaxCrossEntropyWithLogits_grad/mulMul7gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDimsSoftmaxCrossEntropyWithLogits:1*
T0*0
_output_shapes
:������������������
e
gradients/Reshape_grad/ShapeShape	BiasAdd_1*
T0*
out_type0*
_output_shapes
:
�
gradients/Reshape_grad/ReshapeReshape0gradients/SoftmaxCrossEntropyWithLogits_grad/mulgradients/Reshape_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
$gradients/BiasAdd_1_grad/BiasAddGradBiasAddGradgradients/Reshape_grad/Reshape*
T0*
data_formatNHWC*
_output_shapes
:
y
)gradients/BiasAdd_1_grad/tuple/group_depsNoOp^gradients/Reshape_grad/Reshape%^gradients/BiasAdd_1_grad/BiasAddGrad
�
1gradients/BiasAdd_1_grad/tuple/control_dependencyIdentitygradients/Reshape_grad/Reshape*^gradients/BiasAdd_1_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/Reshape_grad/Reshape*'
_output_shapes
:���������
�
3gradients/BiasAdd_1_grad/tuple/control_dependency_1Identity$gradients/BiasAdd_1_grad/BiasAddGrad*^gradients/BiasAdd_1_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients/BiasAdd_1_grad/BiasAddGrad*
_output_shapes
:
�
gradients/MatMul_1_grad/MatMulMatMul1gradients/BiasAdd_1_grad/tuple/control_dependencyVariable_1/read*
transpose_a( *
transpose_b(*
T0*'
_output_shapes
:���������
�
 gradients/MatMul_1_grad/MatMul_1MatMulRelu1gradients/BiasAdd_1_grad/tuple/control_dependency*
transpose_a(*
transpose_b( *
T0*
_output_shapes

:
t
(gradients/MatMul_1_grad/tuple/group_depsNoOp^gradients/MatMul_1_grad/MatMul!^gradients/MatMul_1_grad/MatMul_1
�
0gradients/MatMul_1_grad/tuple/control_dependencyIdentitygradients/MatMul_1_grad/MatMul)^gradients/MatMul_1_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/MatMul_1_grad/MatMul*'
_output_shapes
:���������
�
2gradients/MatMul_1_grad/tuple/control_dependency_1Identity gradients/MatMul_1_grad/MatMul_1)^gradients/MatMul_1_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients/MatMul_1_grad/MatMul_1*
_output_shapes

:
�
gradients/Relu_grad/ReluGradReluGrad0gradients/MatMul_1_grad/tuple/control_dependencyRelu*
T0*'
_output_shapes
:���������
�
"gradients/BiasAdd_grad/BiasAddGradBiasAddGradgradients/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
:
s
'gradients/BiasAdd_grad/tuple/group_depsNoOp^gradients/Relu_grad/ReluGrad#^gradients/BiasAdd_grad/BiasAddGrad
�
/gradients/BiasAdd_grad/tuple/control_dependencyIdentitygradients/Relu_grad/ReluGrad(^gradients/BiasAdd_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/Relu_grad/ReluGrad*'
_output_shapes
:���������
�
1gradients/BiasAdd_grad/tuple/control_dependency_1Identity"gradients/BiasAdd_grad/BiasAddGrad(^gradients/BiasAdd_grad/tuple/group_deps*
T0*5
_class+
)'loc:@gradients/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
�
gradients/MatMul_grad/MatMulMatMul/gradients/BiasAdd_grad/tuple/control_dependencyVariable/read*
transpose_a( *
transpose_b(*
T0*'
_output_shapes
:���������

�
gradients/MatMul_grad/MatMul_1MatMulinputs/gradients/BiasAdd_grad/tuple/control_dependency*
transpose_a(*
transpose_b( *
T0*
_output_shapes

:

n
&gradients/MatMul_grad/tuple/group_depsNoOp^gradients/MatMul_grad/MatMul^gradients/MatMul_grad/MatMul_1
�
.gradients/MatMul_grad/tuple/control_dependencyIdentitygradients/MatMul_grad/MatMul'^gradients/MatMul_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/MatMul_grad/MatMul*'
_output_shapes
:���������

�
0gradients/MatMul_grad/tuple/control_dependency_1Identitygradients/MatMul_grad/MatMul_1'^gradients/MatMul_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/MatMul_grad/MatMul_1*
_output_shapes

:

b
GradientDescent/learning_rateConst*
valueB
 *���=*
dtype0*
_output_shapes
: 
�
4GradientDescent/update_Variable/ApplyGradientDescentApplyGradientDescentVariableGradientDescent/learning_rate0gradients/MatMul_grad/tuple/control_dependency_1*
T0*
use_locking( *
_class
loc:@Variable*
_output_shapes

:

�
6GradientDescent/update_Variable_1/ApplyGradientDescentApplyGradientDescent
Variable_1GradientDescent/learning_rate2gradients/MatMul_1_grad/tuple/control_dependency_1*
T0*
use_locking( *
_class
loc:@Variable_1*
_output_shapes

:
�
6GradientDescent/update_Variable_2/ApplyGradientDescentApplyGradientDescent
Variable_2GradientDescent/learning_rate1gradients/BiasAdd_grad/tuple/control_dependency_1*
T0*
use_locking( *
_class
loc:@Variable_2*
_output_shapes
:
�
6GradientDescent/update_Variable_3/ApplyGradientDescentApplyGradientDescent
Variable_3GradientDescent/learning_rate3gradients/BiasAdd_1_grad/tuple/control_dependency_1*
T0*
use_locking( *
_class
loc:@Variable_3*
_output_shapes
:
�
GradientDescentNoOp5^GradientDescent/update_Variable/ApplyGradientDescent7^GradientDescent/update_Variable_1/ApplyGradientDescent7^GradientDescent/update_Variable_2/ApplyGradientDescent7^GradientDescent/update_Variable_3/ApplyGradientDescent
Z
initNoOp^Variable/Assign^Variable_1/Assign^Variable_2/Assign^Variable_3/Assign""�
	variables��
.

Variable:0Variable/AssignVariable/read:0
4
Variable_1:0Variable_1/AssignVariable_1/read:0
4
Variable_2:0Variable_2/AssignVariable_2/read:0
4
Variable_3:0Variable_3/AssignVariable_3/read:0"�
trainable_variables��
.

Variable:0Variable/AssignVariable/read:0
4
Variable_1:0Variable_1/AssignVariable_1/read:0
4
Variable_2:0Variable_2/AssignVariable_2/read:0
4
Variable_3:0Variable_3/AssignVariable_3/read:0"
train_op

GradientDescent��o