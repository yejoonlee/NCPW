Í&
Õ¦
B
AddV2
x"T
y"T
z"T"
Ttype:
2	
B
AssignVariableOp
resource
value"dtype"
dtypetype
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
.
Identity

input"T
output"T"	
Ttype
:
Less
x"T
y"T
z
"
Ttype:
2	
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
=
Mul
x"T
y"T
z"T"
Ttype:
2	

NoOp
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
@
ReadVariableOp
resource
value"dtype"
dtypetype
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype

SplitV

value"T
size_splits"Tlen
	split_dim
output"T*	num_split"
	num_splitint(0"	
Ttype"
Tlentype0	:
2	
¾
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
@
StaticRegexFullMatch	
input

output
"
patternstring
ö
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
;
Sub
x"T
y"T
z"T"
Ttype:
2	

TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type
output_handle"
element_dtypetype"

shape_typetype:
2	

TensorListReserve
element_shape"
shape_type
num_elements

handle"
element_dtypetype"

shape_typetype:
2	

TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsintÿÿÿÿÿÿÿÿÿ
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
P
Unpack

value"T
output"T*num"
numint("	
Ttype"
axisint 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 

While

input2T
output2T"
T
list(type)("
condfunc"
bodyfunc" 
output_shapeslist(shape)
 "
parallel_iterationsint
"serve*2.4.02v2.4.0-rc4-71-g582c8d236cb8Éì#
t
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense/kernel
m
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes

:*
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0

gru/gru_cell/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:`*$
shared_namegru/gru_cell/kernel
{
'gru/gru_cell/kernel/Read/ReadVariableOpReadVariableOpgru/gru_cell/kernel*
_output_shapes

:`*
dtype0

gru/gru_cell/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: `*.
shared_namegru/gru_cell/recurrent_kernel

1gru/gru_cell/recurrent_kernel/Read/ReadVariableOpReadVariableOpgru/gru_cell/recurrent_kernel*
_output_shapes

: `*
dtype0
~
gru/gru_cell/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:`*"
shared_namegru/gru_cell/bias
w
%gru/gru_cell/bias/Read/ReadVariableOpReadVariableOpgru/gru_cell/bias*
_output_shapes

:`*
dtype0

gru_1/gru_cell_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: 0*(
shared_namegru_1/gru_cell_1/kernel

+gru_1/gru_cell_1/kernel/Read/ReadVariableOpReadVariableOpgru_1/gru_cell_1/kernel*
_output_shapes

: 0*
dtype0

!gru_1/gru_cell_1/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:0*2
shared_name#!gru_1/gru_cell_1/recurrent_kernel

5gru_1/gru_cell_1/recurrent_kernel/Read/ReadVariableOpReadVariableOp!gru_1/gru_cell_1/recurrent_kernel*
_output_shapes

:0*
dtype0

gru_1/gru_cell_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:0*&
shared_namegru_1/gru_cell_1/bias

)gru_1/gru_cell_1/bias/Read/ReadVariableOpReadVariableOpgru_1/gru_cell_1/bias*
_output_shapes

:0*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0

Adam/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*$
shared_nameAdam/dense/kernel/m
{
'Adam/dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/m*
_output_shapes

:*
dtype0
z
Adam/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/dense/bias/m
s
%Adam/dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense/bias/m*
_output_shapes
:*
dtype0

Adam/gru/gru_cell/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:`*+
shared_nameAdam/gru/gru_cell/kernel/m

.Adam/gru/gru_cell/kernel/m/Read/ReadVariableOpReadVariableOpAdam/gru/gru_cell/kernel/m*
_output_shapes

:`*
dtype0
¤
$Adam/gru/gru_cell/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: `*5
shared_name&$Adam/gru/gru_cell/recurrent_kernel/m

8Adam/gru/gru_cell/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp$Adam/gru/gru_cell/recurrent_kernel/m*
_output_shapes

: `*
dtype0

Adam/gru/gru_cell/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:`*)
shared_nameAdam/gru/gru_cell/bias/m

,Adam/gru/gru_cell/bias/m/Read/ReadVariableOpReadVariableOpAdam/gru/gru_cell/bias/m*
_output_shapes

:`*
dtype0

Adam/gru_1/gru_cell_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: 0*/
shared_name Adam/gru_1/gru_cell_1/kernel/m

2Adam/gru_1/gru_cell_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/gru_1/gru_cell_1/kernel/m*
_output_shapes

: 0*
dtype0
¬
(Adam/gru_1/gru_cell_1/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:0*9
shared_name*(Adam/gru_1/gru_cell_1/recurrent_kernel/m
¥
<Adam/gru_1/gru_cell_1/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp(Adam/gru_1/gru_cell_1/recurrent_kernel/m*
_output_shapes

:0*
dtype0

Adam/gru_1/gru_cell_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:0*-
shared_nameAdam/gru_1/gru_cell_1/bias/m

0Adam/gru_1/gru_cell_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/gru_1/gru_cell_1/bias/m*
_output_shapes

:0*
dtype0

Adam/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*$
shared_nameAdam/dense/kernel/v
{
'Adam/dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/v*
_output_shapes

:*
dtype0
z
Adam/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/dense/bias/v
s
%Adam/dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense/bias/v*
_output_shapes
:*
dtype0

Adam/gru/gru_cell/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:`*+
shared_nameAdam/gru/gru_cell/kernel/v

.Adam/gru/gru_cell/kernel/v/Read/ReadVariableOpReadVariableOpAdam/gru/gru_cell/kernel/v*
_output_shapes

:`*
dtype0
¤
$Adam/gru/gru_cell/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: `*5
shared_name&$Adam/gru/gru_cell/recurrent_kernel/v

8Adam/gru/gru_cell/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp$Adam/gru/gru_cell/recurrent_kernel/v*
_output_shapes

: `*
dtype0

Adam/gru/gru_cell/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:`*)
shared_nameAdam/gru/gru_cell/bias/v

,Adam/gru/gru_cell/bias/v/Read/ReadVariableOpReadVariableOpAdam/gru/gru_cell/bias/v*
_output_shapes

:`*
dtype0

Adam/gru_1/gru_cell_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: 0*/
shared_name Adam/gru_1/gru_cell_1/kernel/v

2Adam/gru_1/gru_cell_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/gru_1/gru_cell_1/kernel/v*
_output_shapes

: 0*
dtype0
¬
(Adam/gru_1/gru_cell_1/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:0*9
shared_name*(Adam/gru_1/gru_cell_1/recurrent_kernel/v
¥
<Adam/gru_1/gru_cell_1/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp(Adam/gru_1/gru_cell_1/recurrent_kernel/v*
_output_shapes

:0*
dtype0

Adam/gru_1/gru_cell_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:0*-
shared_nameAdam/gru_1/gru_cell_1/bias/v

0Adam/gru_1/gru_cell_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/gru_1/gru_cell_1/bias/v*
_output_shapes

:0*
dtype0

NoOpNoOp
ú2
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*µ2
value«2B¨2 B¡2

layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
	optimizer
	variables
trainable_variables
	regularization_losses

	keras_api

signatures
l
cell

state_spec
	variables
trainable_variables
regularization_losses
	keras_api
R
	variables
trainable_variables
regularization_losses
	keras_api
l
cell

state_spec
	variables
trainable_variables
regularization_losses
	keras_api
R
	variables
trainable_variables
regularization_losses
	keras_api
h

 kernel
!bias
"	variables
#trainable_variables
$regularization_losses
%	keras_api
Ð
&iter

'beta_1

(beta_2
	)decay
*learning_rate mh!mi+mj,mk-ml.mm/mn0mo vp!vq+vr,vs-vt.vu/vv0vw
8
+0
,1
-2
.3
/4
05
 6
!7
8
+0
,1
-2
.3
/4
05
 6
!7
 
­
	variables
trainable_variables

1layers
2layer_regularization_losses
3non_trainable_variables
4layer_metrics
5metrics
	regularization_losses
 
~

+kernel
,recurrent_kernel
-bias
6	variables
7trainable_variables
8regularization_losses
9	keras_api
 

+0
,1
-2

+0
,1
-2
 
¹
	variables
trainable_variables

:layers
;layer_regularization_losses

<states
=non_trainable_variables
>layer_metrics
?metrics
regularization_losses
 
 
 
­
	variables
trainable_variables

@layers
Alayer_regularization_losses
Bnon_trainable_variables
Clayer_metrics
Dmetrics
regularization_losses
~

.kernel
/recurrent_kernel
0bias
E	variables
Ftrainable_variables
Gregularization_losses
H	keras_api
 

.0
/1
02

.0
/1
02
 
¹
	variables
trainable_variables

Ilayers
Jlayer_regularization_losses

Kstates
Lnon_trainable_variables
Mlayer_metrics
Nmetrics
regularization_losses
 
 
 
­
	variables
trainable_variables

Olayers
Player_regularization_losses
Qnon_trainable_variables
Rlayer_metrics
Smetrics
regularization_losses
XV
VARIABLE_VALUEdense/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
dense/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

 0
!1

 0
!1
 
­
"	variables
#trainable_variables

Tlayers
Ulayer_regularization_losses
Vnon_trainable_variables
Wlayer_metrics
Xmetrics
$regularization_losses
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEgru/gru_cell/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEgru/gru_cell/recurrent_kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEgru/gru_cell/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEgru_1/gru_cell_1/kernel&variables/3/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUE!gru_1/gru_cell_1/recurrent_kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEgru_1/gru_cell_1/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE
#
0
1
2
3
4
 
 
 

Y0

+0
,1
-2

+0
,1
-2
 
­
6	variables
7trainable_variables

Zlayers
[layer_regularization_losses
\non_trainable_variables
]layer_metrics
^metrics
8regularization_losses

0
 
 
 
 
 
 
 
 
 
 

.0
/1
02

.0
/1
02
 
­
E	variables
Ftrainable_variables

_layers
`layer_regularization_losses
anon_trainable_variables
blayer_metrics
cmetrics
Gregularization_losses

0
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
4
	dtotal
	ecount
f	variables
g	keras_api
 
 
 
 
 
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

d0
e1

f	variables
{y
VARIABLE_VALUEAdam/dense/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUEAdam/gru/gru_cell/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE$Adam/gru/gru_cell/recurrent_kernel/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/gru/gru_cell/bias/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/gru_1/gru_cell_1/kernel/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUE(Adam/gru_1/gru_cell_1/recurrent_kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUEAdam/gru_1/gru_cell_1/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUEAdam/gru/gru_cell/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE$Adam/gru/gru_cell/recurrent_kernel/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/gru/gru_cell/bias/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/gru_1/gru_cell_1/kernel/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUE(Adam/gru_1/gru_cell_1/recurrent_kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUEAdam/gru_1/gru_cell_1/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

serving_default_gru_inputPlaceholder*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0* 
shape:ÿÿÿÿÿÿÿÿÿ
ù
StatefulPartitionedCallStatefulPartitionedCallserving_default_gru_inputgru/gru_cell/biasgru/gru_cell/kernelgru/gru_cell/recurrent_kernelgru_1/gru_cell_1/biasgru_1/gru_cell_1/kernel!gru_1/gru_cell_1/recurrent_kerneldense/kernel
dense/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *+
f&R$
"__inference_signature_wrapper_9108
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp'gru/gru_cell/kernel/Read/ReadVariableOp1gru/gru_cell/recurrent_kernel/Read/ReadVariableOp%gru/gru_cell/bias/Read/ReadVariableOp+gru_1/gru_cell_1/kernel/Read/ReadVariableOp5gru_1/gru_cell_1/recurrent_kernel/Read/ReadVariableOp)gru_1/gru_cell_1/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp'Adam/dense/kernel/m/Read/ReadVariableOp%Adam/dense/bias/m/Read/ReadVariableOp.Adam/gru/gru_cell/kernel/m/Read/ReadVariableOp8Adam/gru/gru_cell/recurrent_kernel/m/Read/ReadVariableOp,Adam/gru/gru_cell/bias/m/Read/ReadVariableOp2Adam/gru_1/gru_cell_1/kernel/m/Read/ReadVariableOp<Adam/gru_1/gru_cell_1/recurrent_kernel/m/Read/ReadVariableOp0Adam/gru_1/gru_cell_1/bias/m/Read/ReadVariableOp'Adam/dense/kernel/v/Read/ReadVariableOp%Adam/dense/bias/v/Read/ReadVariableOp.Adam/gru/gru_cell/kernel/v/Read/ReadVariableOp8Adam/gru/gru_cell/recurrent_kernel/v/Read/ReadVariableOp,Adam/gru/gru_cell/bias/v/Read/ReadVariableOp2Adam/gru_1/gru_cell_1/kernel/v/Read/ReadVariableOp<Adam/gru_1/gru_cell_1/recurrent_kernel/v/Read/ReadVariableOp0Adam/gru_1/gru_cell_1/bias/v/Read/ReadVariableOpConst*,
Tin%
#2!	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *'
f"R 
__inference__traced_save_11573

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense/kernel
dense/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rategru/gru_cell/kernelgru/gru_cell/recurrent_kernelgru/gru_cell/biasgru_1/gru_cell_1/kernel!gru_1/gru_cell_1/recurrent_kernelgru_1/gru_cell_1/biastotalcountAdam/dense/kernel/mAdam/dense/bias/mAdam/gru/gru_cell/kernel/m$Adam/gru/gru_cell/recurrent_kernel/mAdam/gru/gru_cell/bias/mAdam/gru_1/gru_cell_1/kernel/m(Adam/gru_1/gru_cell_1/recurrent_kernel/mAdam/gru_1/gru_cell_1/bias/mAdam/dense/kernel/vAdam/dense/bias/vAdam/gru/gru_cell/kernel/v$Adam/gru/gru_cell/recurrent_kernel/vAdam/gru/gru_cell/bias/vAdam/gru_1/gru_cell_1/kernel/v(Adam/gru_1/gru_cell_1/recurrent_kernel/vAdam/gru_1/gru_cell_1/bias/v*+
Tin$
"2 *
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 **
f%R#
!__inference__traced_restore_11676³Þ"
ÏX
Ä
=__inference_gru_layer_call_and_return_conditional_losses_8336

inputs$
 gru_cell_readvariableop_resource+
'gru_cell_matmul_readvariableop_resource-
)gru_cell_matmul_1_readvariableop_resource
identity¢gru_cell/MatMul/ReadVariableOp¢ gru_cell/MatMul_1/ReadVariableOp¢gru_cell/ReadVariableOp¢whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ü
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_2
gru_cell/ReadVariableOpReadVariableOp gru_cell_readvariableop_resource*
_output_shapes

:`*
dtype02
gru_cell/ReadVariableOp
gru_cell/unstackUnpackgru_cell/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num2
gru_cell/unstack¨
gru_cell/MatMul/ReadVariableOpReadVariableOp'gru_cell_matmul_readvariableop_resource*
_output_shapes

:`*
dtype02 
gru_cell/MatMul/ReadVariableOp 
gru_cell/MatMulMatMulstrided_slice_2:output:0&gru_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
gru_cell/MatMul
gru_cell/BiasAddBiasAddgru_cell/MatMul:product:0gru_cell/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
gru_cell/BiasAddb
gru_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru_cell/Const
gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
gru_cell/split/split_dimÐ
gru_cell/splitSplit!gru_cell/split/split_dim:output:0gru_cell/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
gru_cell/split®
 gru_cell/MatMul_1/ReadVariableOpReadVariableOp)gru_cell_matmul_1_readvariableop_resource*
_output_shapes

: `*
dtype02"
 gru_cell/MatMul_1/ReadVariableOp
gru_cell/MatMul_1MatMulzeros:output:0(gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
gru_cell/MatMul_1
gru_cell/BiasAdd_1BiasAddgru_cell/MatMul_1:product:0gru_cell/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
gru_cell/BiasAdd_1y
gru_cell/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"        ÿÿÿÿ2
gru_cell/Const_1
gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
gru_cell/split_1/split_dim
gru_cell/split_1SplitVgru_cell/BiasAdd_1:output:0gru_cell/Const_1:output:0#gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
gru_cell/split_1
gru_cell/addAddV2gru_cell/split:output:0gru_cell/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/adds
gru_cell/SigmoidSigmoidgru_cell/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/Sigmoid
gru_cell/add_1AddV2gru_cell/split:output:1gru_cell/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/add_1y
gru_cell/Sigmoid_1Sigmoidgru_cell/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/Sigmoid_1
gru_cell/mulMulgru_cell/Sigmoid_1:y:0gru_cell/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/mul
gru_cell/add_2AddV2gru_cell/split:output:2gru_cell/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/add_2l
gru_cell/ReluRelugru_cell/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/Relu
gru_cell/mul_1Mulgru_cell/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/mul_1e
gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru_cell/sub/x
gru_cell/subSubgru_cell/sub/x:output:0gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/sub
gru_cell/mul_2Mulgru_cell/sub:z:0gru_cell/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/mul_2
gru_cell/add_3AddV2gru_cell/mul_1:z:0gru_cell/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/add_3
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2
TensorArrayV2_1/element_shape¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0 gru_cell_readvariableop_resource'gru_cell_matmul_readvariableop_resource)gru_cell_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ : : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_8246*
condR
while_cond_8245*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ : : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    22
0TensorArrayV2Stack/TensorListStack/element_shapeè
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¥
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeÍ
IdentityIdentitytranspose_1:y:0^gru_cell/MatMul/ReadVariableOp!^gru_cell/MatMul_1/ReadVariableOp^gru_cell/ReadVariableOp^while*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ:::2@
gru_cell/MatMul/ReadVariableOpgru_cell/MatMul/ReadVariableOp2D
 gru_cell/MatMul_1/ReadVariableOp gru_cell/MatMul_1/ReadVariableOp22
gru_cell/ReadVariableOpgru_cell/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ú

D__inference_sequential_layer_call_and_return_conditional_losses_9012

inputs
gru_8990
gru_8992
gru_8994

gru_1_8998

gru_1_9000

gru_1_9002

dense_9006

dense_9008
identity¢dense/StatefulPartitionedCall¢dropout/StatefulPartitionedCall¢!dropout_1/StatefulPartitionedCall¢gru/StatefulPartitionedCall¢gru_1/StatefulPartitionedCall
gru/StatefulPartitionedCallStatefulPartitionedCallinputsgru_8990gru_8992gru_8994*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *F
fAR?
=__inference_gru_layer_call_and_return_conditional_losses_83362
gru/StatefulPartitionedCall
dropout/StatefulPartitionedCallStatefulPartitionedCall$gru/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_85372!
dropout/StatefulPartitionedCall²
gru_1/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0
gru_1_8998
gru_1_9000
gru_1_9002*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_gru_1_layer_call_and_return_conditional_losses_87132
gru_1/StatefulPartitionedCall²
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall&gru_1/StatefulPartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dropout_1_layer_call_and_return_conditional_losses_89142#
!dropout_1/StatefulPartitionedCall¦
dense/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0
dense_9006
dense_9008*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_89422
dense/StatefulPartitionedCall
IdentityIdentity&dense/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall^gru/StatefulPartitionedCall^gru_1/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:ÿÿÿÿÿÿÿÿÿ::::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2:
gru/StatefulPartitionedCallgru/StatefulPartitionedCall2>
gru_1/StatefulPartitionedCallgru_1/StatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
 
b
)__inference_dropout_1_layer_call_fn_11217

inputs
identity¢StatefulPartitionedCallÜ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dropout_1_layer_call_and_return_conditional_losses_89142
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Y
Æ
=__inference_gru_layer_call_and_return_conditional_losses_9967
inputs_0$
 gru_cell_readvariableop_resource+
'gru_cell_matmul_readvariableop_resource-
)gru_cell_matmul_1_readvariableop_resource
identity¢gru_cell/MatMul/ReadVariableOp¢ gru_cell/MatMul_1/ReadVariableOp¢gru_cell/ReadVariableOp¢whileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ü
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_2
gru_cell/ReadVariableOpReadVariableOp gru_cell_readvariableop_resource*
_output_shapes

:`*
dtype02
gru_cell/ReadVariableOp
gru_cell/unstackUnpackgru_cell/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num2
gru_cell/unstack¨
gru_cell/MatMul/ReadVariableOpReadVariableOp'gru_cell_matmul_readvariableop_resource*
_output_shapes

:`*
dtype02 
gru_cell/MatMul/ReadVariableOp 
gru_cell/MatMulMatMulstrided_slice_2:output:0&gru_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
gru_cell/MatMul
gru_cell/BiasAddBiasAddgru_cell/MatMul:product:0gru_cell/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
gru_cell/BiasAddb
gru_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru_cell/Const
gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
gru_cell/split/split_dimÐ
gru_cell/splitSplit!gru_cell/split/split_dim:output:0gru_cell/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
gru_cell/split®
 gru_cell/MatMul_1/ReadVariableOpReadVariableOp)gru_cell_matmul_1_readvariableop_resource*
_output_shapes

: `*
dtype02"
 gru_cell/MatMul_1/ReadVariableOp
gru_cell/MatMul_1MatMulzeros:output:0(gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
gru_cell/MatMul_1
gru_cell/BiasAdd_1BiasAddgru_cell/MatMul_1:product:0gru_cell/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
gru_cell/BiasAdd_1y
gru_cell/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"        ÿÿÿÿ2
gru_cell/Const_1
gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
gru_cell/split_1/split_dim
gru_cell/split_1SplitVgru_cell/BiasAdd_1:output:0gru_cell/Const_1:output:0#gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
gru_cell/split_1
gru_cell/addAddV2gru_cell/split:output:0gru_cell/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/adds
gru_cell/SigmoidSigmoidgru_cell/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/Sigmoid
gru_cell/add_1AddV2gru_cell/split:output:1gru_cell/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/add_1y
gru_cell/Sigmoid_1Sigmoidgru_cell/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/Sigmoid_1
gru_cell/mulMulgru_cell/Sigmoid_1:y:0gru_cell/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/mul
gru_cell/add_2AddV2gru_cell/split:output:2gru_cell/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/add_2l
gru_cell/ReluRelugru_cell/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/Relu
gru_cell/mul_1Mulgru_cell/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/mul_1e
gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru_cell/sub/x
gru_cell/subSubgru_cell/sub/x:output:0gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/sub
gru_cell/mul_2Mulgru_cell/sub:z:0gru_cell/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/mul_2
gru_cell/add_3AddV2gru_cell/mul_1:z:0gru_cell/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/add_3
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2
TensorArrayV2_1/element_shape¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0 gru_cell_readvariableop_resource'gru_cell_matmul_readvariableop_resource)gru_cell_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ : : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_9877*
condR
while_cond_9876*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ : : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    22
0TensorArrayV2Stack/TensorListStack/element_shapeñ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm®
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeÖ
IdentityIdentitytranspose_1:y:0^gru_cell/MatMul/ReadVariableOp!^gru_cell/MatMul_1/ReadVariableOp^gru_cell/ReadVariableOp^while*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::2@
gru_cell/MatMul/ReadVariableOpgru_cell/MatMul/ReadVariableOp2D
 gru_cell/MatMul_1/ReadVariableOp gru_cell/MatMul_1/ReadVariableOp22
gru_cell/ReadVariableOpgru_cell/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
óF
¤
while_body_10584
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0.
*while_gru_cell_1_readvariableop_resource_05
1while_gru_cell_1_matmul_readvariableop_resource_07
3while_gru_cell_1_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor,
(while_gru_cell_1_readvariableop_resource3
/while_gru_cell_1_matmul_readvariableop_resource5
1while_gru_cell_1_matmul_1_readvariableop_resource¢&while/gru_cell_1/MatMul/ReadVariableOp¢(while/gru_cell_1/MatMul_1/ReadVariableOp¢while/gru_cell_1/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÓ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem­
while/gru_cell_1/ReadVariableOpReadVariableOp*while_gru_cell_1_readvariableop_resource_0*
_output_shapes

:0*
dtype02!
while/gru_cell_1/ReadVariableOp
while/gru_cell_1/unstackUnpack'while/gru_cell_1/ReadVariableOp:value:0*
T0* 
_output_shapes
:0:0*	
num2
while/gru_cell_1/unstackÂ
&while/gru_cell_1/MatMul/ReadVariableOpReadVariableOp1while_gru_cell_1_matmul_readvariableop_resource_0*
_output_shapes

: 0*
dtype02(
&while/gru_cell_1/MatMul/ReadVariableOpÐ
while/gru_cell_1/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/gru_cell_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02
while/gru_cell_1/MatMul·
while/gru_cell_1/BiasAddBiasAdd!while/gru_cell_1/MatMul:product:0!while/gru_cell_1/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02
while/gru_cell_1/BiasAddr
while/gru_cell_1/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/gru_cell_1/Const
 while/gru_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2"
 while/gru_cell_1/split/split_dimð
while/gru_cell_1/splitSplit)while/gru_cell_1/split/split_dim:output:0!while/gru_cell_1/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
while/gru_cell_1/splitÈ
(while/gru_cell_1/MatMul_1/ReadVariableOpReadVariableOp3while_gru_cell_1_matmul_1_readvariableop_resource_0*
_output_shapes

:0*
dtype02*
(while/gru_cell_1/MatMul_1/ReadVariableOp¹
while/gru_cell_1/MatMul_1MatMulwhile_placeholder_20while/gru_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02
while/gru_cell_1/MatMul_1½
while/gru_cell_1/BiasAdd_1BiasAdd#while/gru_cell_1/MatMul_1:product:0!while/gru_cell_1/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02
while/gru_cell_1/BiasAdd_1
while/gru_cell_1/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      ÿÿÿÿ2
while/gru_cell_1/Const_1
"while/gru_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2$
"while/gru_cell_1/split_1/split_dim¨
while/gru_cell_1/split_1SplitV#while/gru_cell_1/BiasAdd_1:output:0!while/gru_cell_1/Const_1:output:0+while/gru_cell_1/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
while/gru_cell_1/split_1«
while/gru_cell_1/addAddV2while/gru_cell_1/split:output:0!while/gru_cell_1/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_1/add
while/gru_cell_1/SigmoidSigmoidwhile/gru_cell_1/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_1/Sigmoid¯
while/gru_cell_1/add_1AddV2while/gru_cell_1/split:output:1!while/gru_cell_1/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_1/add_1
while/gru_cell_1/Sigmoid_1Sigmoidwhile/gru_cell_1/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_1/Sigmoid_1¨
while/gru_cell_1/mulMulwhile/gru_cell_1/Sigmoid_1:y:0!while/gru_cell_1/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_1/mul¦
while/gru_cell_1/add_2AddV2while/gru_cell_1/split:output:2while/gru_cell_1/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_1/add_2
while/gru_cell_1/ReluReluwhile/gru_cell_1/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_1/Relu
while/gru_cell_1/mul_1Mulwhile/gru_cell_1/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_1/mul_1u
while/gru_cell_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
while/gru_cell_1/sub/x¤
while/gru_cell_1/subSubwhile/gru_cell_1/sub/x:output:0while/gru_cell_1/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_1/sub¨
while/gru_cell_1/mul_2Mulwhile/gru_cell_1/sub:z:0#while/gru_cell_1/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_1/mul_2£
while/gru_cell_1/add_3AddV2while/gru_cell_1/mul_1:z:0while/gru_cell_1/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_1/add_3Þ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_1/add_3:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1Ô
while/IdentityIdentitywhile/add_1:z:0'^while/gru_cell_1/MatMul/ReadVariableOp)^while/gru_cell_1/MatMul_1/ReadVariableOp ^while/gru_cell_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identityç
while/Identity_1Identitywhile_while_maximum_iterations'^while/gru_cell_1/MatMul/ReadVariableOp)^while/gru_cell_1/MatMul_1/ReadVariableOp ^while/gru_cell_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1Ö
while/Identity_2Identitywhile/add:z:0'^while/gru_cell_1/MatMul/ReadVariableOp)^while/gru_cell_1/MatMul_1/ReadVariableOp ^while/gru_cell_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0'^while/gru_cell_1/MatMul/ReadVariableOp)^while/gru_cell_1/MatMul_1/ReadVariableOp ^while/gru_cell_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3ô
while/Identity_4Identitywhile/gru_cell_1/add_3:z:0'^while/gru_cell_1/MatMul/ReadVariableOp)^while/gru_cell_1/MatMul_1/ReadVariableOp ^while/gru_cell_1/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_4"h
1while_gru_cell_1_matmul_1_readvariableop_resource3while_gru_cell_1_matmul_1_readvariableop_resource_0"d
/while_gru_cell_1_matmul_readvariableop_resource1while_gru_cell_1_matmul_readvariableop_resource_0"V
(while_gru_cell_1_readvariableop_resource*while_gru_cell_1_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :ÿÿÿÿÿÿÿÿÿ: : :::2P
&while/gru_cell_1/MatMul/ReadVariableOp&while/gru_cell_1/MatMul/ReadVariableOp2T
(while/gru_cell_1/MatMul_1/ReadVariableOp(while/gru_cell_1/MatMul_1/ReadVariableOp2B
while/gru_cell_1/ReadVariableOpwhile/gru_cell_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
ûD

while_body_8405
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0,
(while_gru_cell_readvariableop_resource_03
/while_gru_cell_matmul_readvariableop_resource_05
1while_gru_cell_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor*
&while_gru_cell_readvariableop_resource1
-while_gru_cell_matmul_readvariableop_resource3
/while_gru_cell_matmul_1_readvariableop_resource¢$while/gru_cell/MatMul/ReadVariableOp¢&while/gru_cell/MatMul_1/ReadVariableOp¢while/gru_cell/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÓ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem§
while/gru_cell/ReadVariableOpReadVariableOp(while_gru_cell_readvariableop_resource_0*
_output_shapes

:`*
dtype02
while/gru_cell/ReadVariableOp
while/gru_cell/unstackUnpack%while/gru_cell/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num2
while/gru_cell/unstack¼
$while/gru_cell/MatMul/ReadVariableOpReadVariableOp/while_gru_cell_matmul_readvariableop_resource_0*
_output_shapes

:`*
dtype02&
$while/gru_cell/MatMul/ReadVariableOpÊ
while/gru_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0,while/gru_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
while/gru_cell/MatMul¯
while/gru_cell/BiasAddBiasAddwhile/gru_cell/MatMul:product:0while/gru_cell/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
while/gru_cell/BiasAddn
while/gru_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/gru_cell/Const
while/gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2 
while/gru_cell/split/split_dimè
while/gru_cell/splitSplit'while/gru_cell/split/split_dim:output:0while/gru_cell/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
while/gru_cell/splitÂ
&while/gru_cell/MatMul_1/ReadVariableOpReadVariableOp1while_gru_cell_matmul_1_readvariableop_resource_0*
_output_shapes

: `*
dtype02(
&while/gru_cell/MatMul_1/ReadVariableOp³
while/gru_cell/MatMul_1MatMulwhile_placeholder_2.while/gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
while/gru_cell/MatMul_1µ
while/gru_cell/BiasAdd_1BiasAdd!while/gru_cell/MatMul_1:product:0while/gru_cell/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
while/gru_cell/BiasAdd_1
while/gru_cell/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"        ÿÿÿÿ2
while/gru_cell/Const_1
 while/gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2"
 while/gru_cell/split_1/split_dim
while/gru_cell/split_1SplitV!while/gru_cell/BiasAdd_1:output:0while/gru_cell/Const_1:output:0)while/gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
while/gru_cell/split_1£
while/gru_cell/addAddV2while/gru_cell/split:output:0while/gru_cell/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/add
while/gru_cell/SigmoidSigmoidwhile/gru_cell/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/Sigmoid§
while/gru_cell/add_1AddV2while/gru_cell/split:output:1while/gru_cell/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/add_1
while/gru_cell/Sigmoid_1Sigmoidwhile/gru_cell/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/Sigmoid_1 
while/gru_cell/mulMulwhile/gru_cell/Sigmoid_1:y:0while/gru_cell/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/mul
while/gru_cell/add_2AddV2while/gru_cell/split:output:2while/gru_cell/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/add_2~
while/gru_cell/ReluReluwhile/gru_cell/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/Relu
while/gru_cell/mul_1Mulwhile/gru_cell/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/mul_1q
while/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
while/gru_cell/sub/x
while/gru_cell/subSubwhile/gru_cell/sub/x:output:0while/gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/sub 
while/gru_cell/mul_2Mulwhile/gru_cell/sub:z:0!while/gru_cell/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/mul_2
while/gru_cell/add_3AddV2while/gru_cell/mul_1:z:0while/gru_cell/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/add_3Ü
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell/add_3:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1Î
while/IdentityIdentitywhile/add_1:z:0%^while/gru_cell/MatMul/ReadVariableOp'^while/gru_cell/MatMul_1/ReadVariableOp^while/gru_cell/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identityá
while/Identity_1Identitywhile_while_maximum_iterations%^while/gru_cell/MatMul/ReadVariableOp'^while/gru_cell/MatMul_1/ReadVariableOp^while/gru_cell/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1Ð
while/Identity_2Identitywhile/add:z:0%^while/gru_cell/MatMul/ReadVariableOp'^while/gru_cell/MatMul_1/ReadVariableOp^while/gru_cell/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2ý
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0%^while/gru_cell/MatMul/ReadVariableOp'^while/gru_cell/MatMul_1/ReadVariableOp^while/gru_cell/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3ì
while/Identity_4Identitywhile/gru_cell/add_3:z:0%^while/gru_cell/MatMul/ReadVariableOp'^while/gru_cell/MatMul_1/ReadVariableOp^while/gru_cell/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4"d
/while_gru_cell_matmul_1_readvariableop_resource1while_gru_cell_matmul_1_readvariableop_resource_0"`
-while_gru_cell_matmul_readvariableop_resource/while_gru_cell_matmul_readvariableop_resource_0"R
&while_gru_cell_readvariableop_resource(while_gru_cell_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :ÿÿÿÿÿÿÿÿÿ : : :::2L
$while/gru_cell/MatMul/ReadVariableOp$while/gru_cell/MatMul/ReadVariableOp2P
&while/gru_cell/MatMul_1/ReadVariableOp&while/gru_cell/MatMul_1/ReadVariableOp2>
while/gru_cell/ReadVariableOpwhile/gru_cell/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
: 
óF
¤
while_body_10743
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0.
*while_gru_cell_1_readvariableop_resource_05
1while_gru_cell_1_matmul_readvariableop_resource_07
3while_gru_cell_1_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor,
(while_gru_cell_1_readvariableop_resource3
/while_gru_cell_1_matmul_readvariableop_resource5
1while_gru_cell_1_matmul_1_readvariableop_resource¢&while/gru_cell_1/MatMul/ReadVariableOp¢(while/gru_cell_1/MatMul_1/ReadVariableOp¢while/gru_cell_1/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÓ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem­
while/gru_cell_1/ReadVariableOpReadVariableOp*while_gru_cell_1_readvariableop_resource_0*
_output_shapes

:0*
dtype02!
while/gru_cell_1/ReadVariableOp
while/gru_cell_1/unstackUnpack'while/gru_cell_1/ReadVariableOp:value:0*
T0* 
_output_shapes
:0:0*	
num2
while/gru_cell_1/unstackÂ
&while/gru_cell_1/MatMul/ReadVariableOpReadVariableOp1while_gru_cell_1_matmul_readvariableop_resource_0*
_output_shapes

: 0*
dtype02(
&while/gru_cell_1/MatMul/ReadVariableOpÐ
while/gru_cell_1/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/gru_cell_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02
while/gru_cell_1/MatMul·
while/gru_cell_1/BiasAddBiasAdd!while/gru_cell_1/MatMul:product:0!while/gru_cell_1/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02
while/gru_cell_1/BiasAddr
while/gru_cell_1/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/gru_cell_1/Const
 while/gru_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2"
 while/gru_cell_1/split/split_dimð
while/gru_cell_1/splitSplit)while/gru_cell_1/split/split_dim:output:0!while/gru_cell_1/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
while/gru_cell_1/splitÈ
(while/gru_cell_1/MatMul_1/ReadVariableOpReadVariableOp3while_gru_cell_1_matmul_1_readvariableop_resource_0*
_output_shapes

:0*
dtype02*
(while/gru_cell_1/MatMul_1/ReadVariableOp¹
while/gru_cell_1/MatMul_1MatMulwhile_placeholder_20while/gru_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02
while/gru_cell_1/MatMul_1½
while/gru_cell_1/BiasAdd_1BiasAdd#while/gru_cell_1/MatMul_1:product:0!while/gru_cell_1/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02
while/gru_cell_1/BiasAdd_1
while/gru_cell_1/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      ÿÿÿÿ2
while/gru_cell_1/Const_1
"while/gru_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2$
"while/gru_cell_1/split_1/split_dim¨
while/gru_cell_1/split_1SplitV#while/gru_cell_1/BiasAdd_1:output:0!while/gru_cell_1/Const_1:output:0+while/gru_cell_1/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
while/gru_cell_1/split_1«
while/gru_cell_1/addAddV2while/gru_cell_1/split:output:0!while/gru_cell_1/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_1/add
while/gru_cell_1/SigmoidSigmoidwhile/gru_cell_1/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_1/Sigmoid¯
while/gru_cell_1/add_1AddV2while/gru_cell_1/split:output:1!while/gru_cell_1/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_1/add_1
while/gru_cell_1/Sigmoid_1Sigmoidwhile/gru_cell_1/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_1/Sigmoid_1¨
while/gru_cell_1/mulMulwhile/gru_cell_1/Sigmoid_1:y:0!while/gru_cell_1/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_1/mul¦
while/gru_cell_1/add_2AddV2while/gru_cell_1/split:output:2while/gru_cell_1/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_1/add_2
while/gru_cell_1/ReluReluwhile/gru_cell_1/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_1/Relu
while/gru_cell_1/mul_1Mulwhile/gru_cell_1/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_1/mul_1u
while/gru_cell_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
while/gru_cell_1/sub/x¤
while/gru_cell_1/subSubwhile/gru_cell_1/sub/x:output:0while/gru_cell_1/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_1/sub¨
while/gru_cell_1/mul_2Mulwhile/gru_cell_1/sub:z:0#while/gru_cell_1/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_1/mul_2£
while/gru_cell_1/add_3AddV2while/gru_cell_1/mul_1:z:0while/gru_cell_1/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_1/add_3Þ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_1/add_3:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1Ô
while/IdentityIdentitywhile/add_1:z:0'^while/gru_cell_1/MatMul/ReadVariableOp)^while/gru_cell_1/MatMul_1/ReadVariableOp ^while/gru_cell_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identityç
while/Identity_1Identitywhile_while_maximum_iterations'^while/gru_cell_1/MatMul/ReadVariableOp)^while/gru_cell_1/MatMul_1/ReadVariableOp ^while/gru_cell_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1Ö
while/Identity_2Identitywhile/add:z:0'^while/gru_cell_1/MatMul/ReadVariableOp)^while/gru_cell_1/MatMul_1/ReadVariableOp ^while/gru_cell_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0'^while/gru_cell_1/MatMul/ReadVariableOp)^while/gru_cell_1/MatMul_1/ReadVariableOp ^while/gru_cell_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3ô
while/Identity_4Identitywhile/gru_cell_1/add_3:z:0'^while/gru_cell_1/MatMul/ReadVariableOp)^while/gru_cell_1/MatMul_1/ReadVariableOp ^while/gru_cell_1/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_4"h
1while_gru_cell_1_matmul_1_readvariableop_resource3while_gru_cell_1_matmul_1_readvariableop_resource_0"d
/while_gru_cell_1_matmul_readvariableop_resource1while_gru_cell_1_matmul_readvariableop_resource_0"V
(while_gru_cell_1_readvariableop_resource*while_gru_cell_1_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :ÿÿÿÿÿÿÿÿÿ: : :::2P
&while/gru_cell_1/MatMul/ReadVariableOp&while/gru_cell_1/MatMul/ReadVariableOp2T
(while/gru_cell_1/MatMul_1/ReadVariableOp(while/gru_cell_1/MatMul_1/ReadVariableOp2B
while/gru_cell_1/ReadVariableOpwhile/gru_cell_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
ßZ
Õ
@__inference_gru_1_layer_call_and_return_conditional_losses_11173
inputs_0&
"gru_cell_1_readvariableop_resource-
)gru_cell_1_matmul_readvariableop_resource/
+gru_cell_1_matmul_1_readvariableop_resource
identity¢ gru_cell_1/MatMul/ReadVariableOp¢"gru_cell_1/MatMul_1/ReadVariableOp¢gru_cell_1/ReadVariableOp¢whileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ü
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2
strided_slice_2
gru_cell_1/ReadVariableOpReadVariableOp"gru_cell_1_readvariableop_resource*
_output_shapes

:0*
dtype02
gru_cell_1/ReadVariableOp
gru_cell_1/unstackUnpack!gru_cell_1/ReadVariableOp:value:0*
T0* 
_output_shapes
:0:0*	
num2
gru_cell_1/unstack®
 gru_cell_1/MatMul/ReadVariableOpReadVariableOp)gru_cell_1_matmul_readvariableop_resource*
_output_shapes

: 0*
dtype02"
 gru_cell_1/MatMul/ReadVariableOp¦
gru_cell_1/MatMulMatMulstrided_slice_2:output:0(gru_cell_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02
gru_cell_1/MatMul
gru_cell_1/BiasAddBiasAddgru_cell_1/MatMul:product:0gru_cell_1/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02
gru_cell_1/BiasAddf
gru_cell_1/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru_cell_1/Const
gru_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
gru_cell_1/split/split_dimØ
gru_cell_1/splitSplit#gru_cell_1/split/split_dim:output:0gru_cell_1/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
gru_cell_1/split´
"gru_cell_1/MatMul_1/ReadVariableOpReadVariableOp+gru_cell_1_matmul_1_readvariableop_resource*
_output_shapes

:0*
dtype02$
"gru_cell_1/MatMul_1/ReadVariableOp¢
gru_cell_1/MatMul_1MatMulzeros:output:0*gru_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02
gru_cell_1/MatMul_1¥
gru_cell_1/BiasAdd_1BiasAddgru_cell_1/MatMul_1:product:0gru_cell_1/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02
gru_cell_1/BiasAdd_1}
gru_cell_1/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      ÿÿÿÿ2
gru_cell_1/Const_1
gru_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
gru_cell_1/split_1/split_dim
gru_cell_1/split_1SplitVgru_cell_1/BiasAdd_1:output:0gru_cell_1/Const_1:output:0%gru_cell_1/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
gru_cell_1/split_1
gru_cell_1/addAddV2gru_cell_1/split:output:0gru_cell_1/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_cell_1/addy
gru_cell_1/SigmoidSigmoidgru_cell_1/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_cell_1/Sigmoid
gru_cell_1/add_1AddV2gru_cell_1/split:output:1gru_cell_1/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_cell_1/add_1
gru_cell_1/Sigmoid_1Sigmoidgru_cell_1/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_cell_1/Sigmoid_1
gru_cell_1/mulMulgru_cell_1/Sigmoid_1:y:0gru_cell_1/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_cell_1/mul
gru_cell_1/add_2AddV2gru_cell_1/split:output:2gru_cell_1/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_cell_1/add_2r
gru_cell_1/ReluRelugru_cell_1/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_cell_1/Relu
gru_cell_1/mul_1Mulgru_cell_1/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_cell_1/mul_1i
gru_cell_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru_cell_1/sub/x
gru_cell_1/subSubgru_cell_1/sub/x:output:0gru_cell_1/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_cell_1/sub
gru_cell_1/mul_2Mulgru_cell_1/sub:z:0gru_cell_1/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_cell_1/mul_2
gru_cell_1/add_3AddV2gru_cell_1/mul_1:z:0gru_cell_1/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_cell_1/add_3
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
TensorArrayV2_1/element_shape¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter¥
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0"gru_cell_1_readvariableop_resource)gru_cell_1_matmul_readvariableop_resource+gru_cell_1_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_11083*
condR
while_cond_11082*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   22
0TensorArrayV2Stack/TensorListStack/element_shapeñ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm®
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeØ
IdentityIdentitystrided_slice_3:output:0!^gru_cell_1/MatMul/ReadVariableOp#^gru_cell_1/MatMul_1/ReadVariableOp^gru_cell_1/ReadVariableOp^while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ :::2D
 gru_cell_1/MatMul/ReadVariableOp gru_cell_1/MatMul/ReadVariableOp2H
"gru_cell_1/MatMul_1/ReadVariableOp"gru_cell_1/MatMul_1/ReadVariableOp26
gru_cell_1/ReadVariableOpgru_cell_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
inputs/0
à
»
D__inference_sequential_layer_call_and_return_conditional_losses_9058

inputs
gru_9036
gru_9038
gru_9040

gru_1_9044

gru_1_9046

gru_1_9048

dense_9052

dense_9054
identity¢dense/StatefulPartitionedCall¢gru/StatefulPartitionedCall¢gru_1/StatefulPartitionedCall
gru/StatefulPartitionedCallStatefulPartitionedCallinputsgru_9036gru_9038gru_9040*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *F
fAR?
=__inference_gru_layer_call_and_return_conditional_losses_84952
gru/StatefulPartitionedCallô
dropout/PartitionedCallPartitionedCall$gru/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_85422
dropout/PartitionedCallª
gru_1/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0
gru_1_9044
gru_1_9046
gru_1_9048*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_gru_1_layer_call_and_return_conditional_losses_88722
gru_1/StatefulPartitionedCallø
dropout_1/PartitionedCallPartitionedCall&gru_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dropout_1_layer_call_and_return_conditional_losses_89192
dropout_1/PartitionedCall
dense/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0
dense_9052
dense_9054*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_89422
dense/StatefulPartitionedCallØ
IdentityIdentity&dense/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall^gru/StatefulPartitionedCall^gru_1/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:ÿÿÿÿÿÿÿÿÿ::::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2:
gru/StatefulPartitionedCallgru/StatefulPartitionedCall2>
gru_1/StatefulPartitionedCallgru_1/StatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ë
¥
while_cond_10035
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_10035___redundant_placeholder03
/while_while_cond_10035___redundant_placeholder13
/while_while_cond_10035___redundant_placeholder23
/while_while_cond_10035___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
:
¨
Ø
)__inference_sequential_layer_call_fn_9808

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity¢StatefulPartitionedCallÅ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_90582
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:ÿÿÿÿÿÿÿÿÿ::::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
	
Ù
@__inference_dense_layer_call_and_return_conditional_losses_11232

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
©Z
Ó
@__inference_gru_1_layer_call_and_return_conditional_losses_10674

inputs&
"gru_cell_1_readvariableop_resource-
)gru_cell_1_matmul_readvariableop_resource/
+gru_cell_1_matmul_1_readvariableop_resource
identity¢ gru_cell_1/MatMul/ReadVariableOp¢"gru_cell_1/MatMul_1/ReadVariableOp¢gru_cell_1/ReadVariableOp¢whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ü
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2
strided_slice_2
gru_cell_1/ReadVariableOpReadVariableOp"gru_cell_1_readvariableop_resource*
_output_shapes

:0*
dtype02
gru_cell_1/ReadVariableOp
gru_cell_1/unstackUnpack!gru_cell_1/ReadVariableOp:value:0*
T0* 
_output_shapes
:0:0*	
num2
gru_cell_1/unstack®
 gru_cell_1/MatMul/ReadVariableOpReadVariableOp)gru_cell_1_matmul_readvariableop_resource*
_output_shapes

: 0*
dtype02"
 gru_cell_1/MatMul/ReadVariableOp¦
gru_cell_1/MatMulMatMulstrided_slice_2:output:0(gru_cell_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02
gru_cell_1/MatMul
gru_cell_1/BiasAddBiasAddgru_cell_1/MatMul:product:0gru_cell_1/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02
gru_cell_1/BiasAddf
gru_cell_1/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru_cell_1/Const
gru_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
gru_cell_1/split/split_dimØ
gru_cell_1/splitSplit#gru_cell_1/split/split_dim:output:0gru_cell_1/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
gru_cell_1/split´
"gru_cell_1/MatMul_1/ReadVariableOpReadVariableOp+gru_cell_1_matmul_1_readvariableop_resource*
_output_shapes

:0*
dtype02$
"gru_cell_1/MatMul_1/ReadVariableOp¢
gru_cell_1/MatMul_1MatMulzeros:output:0*gru_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02
gru_cell_1/MatMul_1¥
gru_cell_1/BiasAdd_1BiasAddgru_cell_1/MatMul_1:product:0gru_cell_1/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02
gru_cell_1/BiasAdd_1}
gru_cell_1/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      ÿÿÿÿ2
gru_cell_1/Const_1
gru_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
gru_cell_1/split_1/split_dim
gru_cell_1/split_1SplitVgru_cell_1/BiasAdd_1:output:0gru_cell_1/Const_1:output:0%gru_cell_1/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
gru_cell_1/split_1
gru_cell_1/addAddV2gru_cell_1/split:output:0gru_cell_1/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_cell_1/addy
gru_cell_1/SigmoidSigmoidgru_cell_1/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_cell_1/Sigmoid
gru_cell_1/add_1AddV2gru_cell_1/split:output:1gru_cell_1/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_cell_1/add_1
gru_cell_1/Sigmoid_1Sigmoidgru_cell_1/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_cell_1/Sigmoid_1
gru_cell_1/mulMulgru_cell_1/Sigmoid_1:y:0gru_cell_1/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_cell_1/mul
gru_cell_1/add_2AddV2gru_cell_1/split:output:2gru_cell_1/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_cell_1/add_2r
gru_cell_1/ReluRelugru_cell_1/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_cell_1/Relu
gru_cell_1/mul_1Mulgru_cell_1/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_cell_1/mul_1i
gru_cell_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru_cell_1/sub/x
gru_cell_1/subSubgru_cell_1/sub/x:output:0gru_cell_1/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_cell_1/sub
gru_cell_1/mul_2Mulgru_cell_1/sub:z:0gru_cell_1/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_cell_1/mul_2
gru_cell_1/add_3AddV2gru_cell_1/mul_1:z:0gru_cell_1/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_cell_1/add_3
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
TensorArrayV2_1/element_shape¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter¥
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0"gru_cell_1_readvariableop_resource)gru_cell_1_matmul_readvariableop_resource+gru_cell_1_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_10584*
condR
while_cond_10583*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   22
0TensorArrayV2Stack/TensorListStack/element_shapeè
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¥
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeØ
IdentityIdentitystrided_slice_3:output:0!^gru_cell_1/MatMul/ReadVariableOp#^gru_cell_1/MatMul_1/ReadVariableOp^gru_cell_1/ReadVariableOp^while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ :::2D
 gru_cell_1/MatMul/ReadVariableOp gru_cell_1/MatMul/ReadVariableOp2H
"gru_cell_1/MatMul_1/ReadVariableOp"gru_cell_1/MatMul_1/ReadVariableOp26
gru_cell_1/ReadVariableOpgru_cell_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ÿ

%__inference_gru_1_layer_call_fn_10844

inputs
unknown
	unknown_0
	unknown_1
identity¢StatefulPartitionedCallÿ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_gru_1_layer_call_and_return_conditional_losses_87132
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ :::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
¾<
Ê
?__inference_gru_1_layer_call_and_return_conditional_losses_8165

inputs
gru_cell_1_8089
gru_cell_1_8091
gru_cell_1_8093
identity¢"gru_cell_1/StatefulPartitionedCall¢whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ü
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2
strided_slice_2å
"gru_cell_1/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0gru_cell_1_8089gru_cell_1_8091gru_cell_1_8093*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_gru_cell_1_layer_call_and_return_conditional_losses_77242$
"gru_cell_1/StatefulPartitionedCall
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
TensorArrayV2_1/element_shape¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterÚ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0gru_cell_1_8089gru_cell_1_8091gru_cell_1_8093*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_8101*
condR
while_cond_8100*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   22
0TensorArrayV2Stack/TensorListStack/element_shapeñ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm®
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime
IdentityIdentitystrided_slice_3:output:0#^gru_cell_1/StatefulPartitionedCall^while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ :::2H
"gru_cell_1/StatefulPartitionedCall"gru_cell_1/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
 
C
'__inference_dropout_layer_call_fn_10515

inputs
identityÆ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_85422
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
óF
¤
while_body_10924
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0.
*while_gru_cell_1_readvariableop_resource_05
1while_gru_cell_1_matmul_readvariableop_resource_07
3while_gru_cell_1_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor,
(while_gru_cell_1_readvariableop_resource3
/while_gru_cell_1_matmul_readvariableop_resource5
1while_gru_cell_1_matmul_1_readvariableop_resource¢&while/gru_cell_1/MatMul/ReadVariableOp¢(while/gru_cell_1/MatMul_1/ReadVariableOp¢while/gru_cell_1/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÓ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem­
while/gru_cell_1/ReadVariableOpReadVariableOp*while_gru_cell_1_readvariableop_resource_0*
_output_shapes

:0*
dtype02!
while/gru_cell_1/ReadVariableOp
while/gru_cell_1/unstackUnpack'while/gru_cell_1/ReadVariableOp:value:0*
T0* 
_output_shapes
:0:0*	
num2
while/gru_cell_1/unstackÂ
&while/gru_cell_1/MatMul/ReadVariableOpReadVariableOp1while_gru_cell_1_matmul_readvariableop_resource_0*
_output_shapes

: 0*
dtype02(
&while/gru_cell_1/MatMul/ReadVariableOpÐ
while/gru_cell_1/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/gru_cell_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02
while/gru_cell_1/MatMul·
while/gru_cell_1/BiasAddBiasAdd!while/gru_cell_1/MatMul:product:0!while/gru_cell_1/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02
while/gru_cell_1/BiasAddr
while/gru_cell_1/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/gru_cell_1/Const
 while/gru_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2"
 while/gru_cell_1/split/split_dimð
while/gru_cell_1/splitSplit)while/gru_cell_1/split/split_dim:output:0!while/gru_cell_1/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
while/gru_cell_1/splitÈ
(while/gru_cell_1/MatMul_1/ReadVariableOpReadVariableOp3while_gru_cell_1_matmul_1_readvariableop_resource_0*
_output_shapes

:0*
dtype02*
(while/gru_cell_1/MatMul_1/ReadVariableOp¹
while/gru_cell_1/MatMul_1MatMulwhile_placeholder_20while/gru_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02
while/gru_cell_1/MatMul_1½
while/gru_cell_1/BiasAdd_1BiasAdd#while/gru_cell_1/MatMul_1:product:0!while/gru_cell_1/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02
while/gru_cell_1/BiasAdd_1
while/gru_cell_1/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      ÿÿÿÿ2
while/gru_cell_1/Const_1
"while/gru_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2$
"while/gru_cell_1/split_1/split_dim¨
while/gru_cell_1/split_1SplitV#while/gru_cell_1/BiasAdd_1:output:0!while/gru_cell_1/Const_1:output:0+while/gru_cell_1/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
while/gru_cell_1/split_1«
while/gru_cell_1/addAddV2while/gru_cell_1/split:output:0!while/gru_cell_1/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_1/add
while/gru_cell_1/SigmoidSigmoidwhile/gru_cell_1/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_1/Sigmoid¯
while/gru_cell_1/add_1AddV2while/gru_cell_1/split:output:1!while/gru_cell_1/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_1/add_1
while/gru_cell_1/Sigmoid_1Sigmoidwhile/gru_cell_1/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_1/Sigmoid_1¨
while/gru_cell_1/mulMulwhile/gru_cell_1/Sigmoid_1:y:0!while/gru_cell_1/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_1/mul¦
while/gru_cell_1/add_2AddV2while/gru_cell_1/split:output:2while/gru_cell_1/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_1/add_2
while/gru_cell_1/ReluReluwhile/gru_cell_1/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_1/Relu
while/gru_cell_1/mul_1Mulwhile/gru_cell_1/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_1/mul_1u
while/gru_cell_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
while/gru_cell_1/sub/x¤
while/gru_cell_1/subSubwhile/gru_cell_1/sub/x:output:0while/gru_cell_1/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_1/sub¨
while/gru_cell_1/mul_2Mulwhile/gru_cell_1/sub:z:0#while/gru_cell_1/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_1/mul_2£
while/gru_cell_1/add_3AddV2while/gru_cell_1/mul_1:z:0while/gru_cell_1/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_1/add_3Þ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_1/add_3:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1Ô
while/IdentityIdentitywhile/add_1:z:0'^while/gru_cell_1/MatMul/ReadVariableOp)^while/gru_cell_1/MatMul_1/ReadVariableOp ^while/gru_cell_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identityç
while/Identity_1Identitywhile_while_maximum_iterations'^while/gru_cell_1/MatMul/ReadVariableOp)^while/gru_cell_1/MatMul_1/ReadVariableOp ^while/gru_cell_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1Ö
while/Identity_2Identitywhile/add:z:0'^while/gru_cell_1/MatMul/ReadVariableOp)^while/gru_cell_1/MatMul_1/ReadVariableOp ^while/gru_cell_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0'^while/gru_cell_1/MatMul/ReadVariableOp)^while/gru_cell_1/MatMul_1/ReadVariableOp ^while/gru_cell_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3ô
while/Identity_4Identitywhile/gru_cell_1/add_3:z:0'^while/gru_cell_1/MatMul/ReadVariableOp)^while/gru_cell_1/MatMul_1/ReadVariableOp ^while/gru_cell_1/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_4"h
1while_gru_cell_1_matmul_1_readvariableop_resource3while_gru_cell_1_matmul_1_readvariableop_resource_0"d
/while_gru_cell_1_matmul_readvariableop_resource1while_gru_cell_1_matmul_readvariableop_resource_0"V
(while_gru_cell_1_readvariableop_resource*while_gru_cell_1_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :ÿÿÿÿÿÿÿÿÿ: : :::2P
&while/gru_cell_1/MatMul/ReadVariableOp&while/gru_cell_1/MatMul/ReadVariableOp2T
(while/gru_cell_1/MatMul_1/ReadVariableOp(while/gru_cell_1/MatMul_1/ReadVariableOp2B
while/gru_cell_1/ReadVariableOpwhile/gru_cell_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
Ü	
¬
*__inference_gru_cell_1_layer_call_fn_11443

inputs
states_0
unknown
	unknown_0
	unknown_1
identity

identity_1¢StatefulPartitionedCall£
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_gru_cell_1_layer_call_and_return_conditional_losses_76842
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*E
_input_shapes4
2:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ:::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/0
Õ
`
B__inference_dropout_layer_call_and_return_conditional_losses_10505

inputs

identity_1^
IdentityIdentityinputs*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identitym

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity_1"!

identity_1Identity_1:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Ô
_
A__inference_dropout_layer_call_and_return_conditional_losses_8542

inputs

identity_1^
IdentityIdentityinputs*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identitym

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity_1"!

identity_1Identity_1:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Æ
 
while_cond_8245
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_12
.while_while_cond_8245___redundant_placeholder02
.while_while_cond_8245___redundant_placeholder12
.while_while_cond_8245___redundant_placeholder22
.while_while_cond_8245___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
:
»
ì
gru_while_cond_9176$
 gru_while_gru_while_loop_counter*
&gru_while_gru_while_maximum_iterations
gru_while_placeholder
gru_while_placeholder_1
gru_while_placeholder_2&
"gru_while_less_gru_strided_slice_1:
6gru_while_gru_while_cond_9176___redundant_placeholder0:
6gru_while_gru_while_cond_9176___redundant_placeholder1:
6gru_while_gru_while_cond_9176___redundant_placeholder2:
6gru_while_gru_while_cond_9176___redundant_placeholder3
gru_while_identity

gru/while/LessLessgru_while_placeholder"gru_while_less_gru_strided_slice_1*
T0*
_output_shapes
: 2
gru/while/Lessi
gru/while/IdentityIdentitygru/while/Less:z:0*
T0
*
_output_shapes
: 2
gru/while/Identity"1
gru_while_identitygru/while/Identity:output:0*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
:
¬
`
'__inference_dropout_layer_call_fn_10510

inputs
identity¢StatefulPartitionedCallÞ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_85372
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Ö
z
%__inference_dense_layer_call_fn_11241

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallò
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_89422
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
©Z
Ó
@__inference_gru_1_layer_call_and_return_conditional_losses_10833

inputs&
"gru_cell_1_readvariableop_resource-
)gru_cell_1_matmul_readvariableop_resource/
+gru_cell_1_matmul_1_readvariableop_resource
identity¢ gru_cell_1/MatMul/ReadVariableOp¢"gru_cell_1/MatMul_1/ReadVariableOp¢gru_cell_1/ReadVariableOp¢whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ü
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2
strided_slice_2
gru_cell_1/ReadVariableOpReadVariableOp"gru_cell_1_readvariableop_resource*
_output_shapes

:0*
dtype02
gru_cell_1/ReadVariableOp
gru_cell_1/unstackUnpack!gru_cell_1/ReadVariableOp:value:0*
T0* 
_output_shapes
:0:0*	
num2
gru_cell_1/unstack®
 gru_cell_1/MatMul/ReadVariableOpReadVariableOp)gru_cell_1_matmul_readvariableop_resource*
_output_shapes

: 0*
dtype02"
 gru_cell_1/MatMul/ReadVariableOp¦
gru_cell_1/MatMulMatMulstrided_slice_2:output:0(gru_cell_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02
gru_cell_1/MatMul
gru_cell_1/BiasAddBiasAddgru_cell_1/MatMul:product:0gru_cell_1/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02
gru_cell_1/BiasAddf
gru_cell_1/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru_cell_1/Const
gru_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
gru_cell_1/split/split_dimØ
gru_cell_1/splitSplit#gru_cell_1/split/split_dim:output:0gru_cell_1/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
gru_cell_1/split´
"gru_cell_1/MatMul_1/ReadVariableOpReadVariableOp+gru_cell_1_matmul_1_readvariableop_resource*
_output_shapes

:0*
dtype02$
"gru_cell_1/MatMul_1/ReadVariableOp¢
gru_cell_1/MatMul_1MatMulzeros:output:0*gru_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02
gru_cell_1/MatMul_1¥
gru_cell_1/BiasAdd_1BiasAddgru_cell_1/MatMul_1:product:0gru_cell_1/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02
gru_cell_1/BiasAdd_1}
gru_cell_1/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      ÿÿÿÿ2
gru_cell_1/Const_1
gru_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
gru_cell_1/split_1/split_dim
gru_cell_1/split_1SplitVgru_cell_1/BiasAdd_1:output:0gru_cell_1/Const_1:output:0%gru_cell_1/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
gru_cell_1/split_1
gru_cell_1/addAddV2gru_cell_1/split:output:0gru_cell_1/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_cell_1/addy
gru_cell_1/SigmoidSigmoidgru_cell_1/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_cell_1/Sigmoid
gru_cell_1/add_1AddV2gru_cell_1/split:output:1gru_cell_1/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_cell_1/add_1
gru_cell_1/Sigmoid_1Sigmoidgru_cell_1/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_cell_1/Sigmoid_1
gru_cell_1/mulMulgru_cell_1/Sigmoid_1:y:0gru_cell_1/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_cell_1/mul
gru_cell_1/add_2AddV2gru_cell_1/split:output:2gru_cell_1/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_cell_1/add_2r
gru_cell_1/ReluRelugru_cell_1/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_cell_1/Relu
gru_cell_1/mul_1Mulgru_cell_1/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_cell_1/mul_1i
gru_cell_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru_cell_1/sub/x
gru_cell_1/subSubgru_cell_1/sub/x:output:0gru_cell_1/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_cell_1/sub
gru_cell_1/mul_2Mulgru_cell_1/sub:z:0gru_cell_1/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_cell_1/mul_2
gru_cell_1/add_3AddV2gru_cell_1/mul_1:z:0gru_cell_1/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_cell_1/add_3
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
TensorArrayV2_1/element_shape¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter¥
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0"gru_cell_1_readvariableop_resource)gru_cell_1_matmul_readvariableop_resource+gru_cell_1_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_10743*
condR
while_cond_10742*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   22
0TensorArrayV2Stack/TensorListStack/element_shapeè
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¥
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeØ
IdentityIdentitystrided_slice_3:output:0!^gru_cell_1/MatMul/ReadVariableOp#^gru_cell_1/MatMul_1/ReadVariableOp^gru_cell_1/ReadVariableOp^while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ :::2D
 gru_cell_1/MatMul/ReadVariableOp gru_cell_1/MatMul/ReadVariableOp2H
"gru_cell_1/MatMul_1/ReadVariableOp"gru_cell_1/MatMul_1/ReadVariableOp26
gru_cell_1/ReadVariableOpgru_cell_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ÒX
Å
>__inference_gru_layer_call_and_return_conditional_losses_10307

inputs$
 gru_cell_readvariableop_resource+
'gru_cell_matmul_readvariableop_resource-
)gru_cell_matmul_1_readvariableop_resource
identity¢gru_cell/MatMul/ReadVariableOp¢ gru_cell/MatMul_1/ReadVariableOp¢gru_cell/ReadVariableOp¢whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ü
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_2
gru_cell/ReadVariableOpReadVariableOp gru_cell_readvariableop_resource*
_output_shapes

:`*
dtype02
gru_cell/ReadVariableOp
gru_cell/unstackUnpackgru_cell/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num2
gru_cell/unstack¨
gru_cell/MatMul/ReadVariableOpReadVariableOp'gru_cell_matmul_readvariableop_resource*
_output_shapes

:`*
dtype02 
gru_cell/MatMul/ReadVariableOp 
gru_cell/MatMulMatMulstrided_slice_2:output:0&gru_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
gru_cell/MatMul
gru_cell/BiasAddBiasAddgru_cell/MatMul:product:0gru_cell/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
gru_cell/BiasAddb
gru_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru_cell/Const
gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
gru_cell/split/split_dimÐ
gru_cell/splitSplit!gru_cell/split/split_dim:output:0gru_cell/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
gru_cell/split®
 gru_cell/MatMul_1/ReadVariableOpReadVariableOp)gru_cell_matmul_1_readvariableop_resource*
_output_shapes

: `*
dtype02"
 gru_cell/MatMul_1/ReadVariableOp
gru_cell/MatMul_1MatMulzeros:output:0(gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
gru_cell/MatMul_1
gru_cell/BiasAdd_1BiasAddgru_cell/MatMul_1:product:0gru_cell/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
gru_cell/BiasAdd_1y
gru_cell/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"        ÿÿÿÿ2
gru_cell/Const_1
gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
gru_cell/split_1/split_dim
gru_cell/split_1SplitVgru_cell/BiasAdd_1:output:0gru_cell/Const_1:output:0#gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
gru_cell/split_1
gru_cell/addAddV2gru_cell/split:output:0gru_cell/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/adds
gru_cell/SigmoidSigmoidgru_cell/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/Sigmoid
gru_cell/add_1AddV2gru_cell/split:output:1gru_cell/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/add_1y
gru_cell/Sigmoid_1Sigmoidgru_cell/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/Sigmoid_1
gru_cell/mulMulgru_cell/Sigmoid_1:y:0gru_cell/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/mul
gru_cell/add_2AddV2gru_cell/split:output:2gru_cell/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/add_2l
gru_cell/ReluRelugru_cell/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/Relu
gru_cell/mul_1Mulgru_cell/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/mul_1e
gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru_cell/sub/x
gru_cell/subSubgru_cell/sub/x:output:0gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/sub
gru_cell/mul_2Mulgru_cell/sub:z:0gru_cell/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/mul_2
gru_cell/add_3AddV2gru_cell/mul_1:z:0gru_cell/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/add_3
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2
TensorArrayV2_1/element_shape¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0 gru_cell_readvariableop_resource'gru_cell_matmul_readvariableop_resource)gru_cell_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ : : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_10217*
condR
while_cond_10216*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ : : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    22
0TensorArrayV2Stack/TensorListStack/element_shapeè
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¥
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeÍ
IdentityIdentitytranspose_1:y:0^gru_cell/MatMul/ReadVariableOp!^gru_cell/MatMul_1/ReadVariableOp^gru_cell/ReadVariableOp^while*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ:::2@
gru_cell/MatMul/ReadVariableOpgru_cell/MatMul/ReadVariableOp2D
 gru_cell/MatMul_1/ReadVariableOp gru_cell/MatMul_1/ReadVariableOp22
gru_cell/ReadVariableOpgru_cell/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¾
©
D__inference_gru_cell_1_layer_call_and_return_conditional_losses_7724

inputs

states
readvariableop_resource"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource
identity

identity_1¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp¢ReadVariableOpx
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:0*
dtype02
ReadVariableOpj
unstackUnpackReadVariableOp:value:0*
T0* 
_output_shapes
:0:0*	
num2	
unstack
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: 0*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02
MatMuls
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constm
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
split/split_dim¬
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
split
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:0*
dtype02
MatMul_1/ReadVariableOpy
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02

MatMul_1y
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02
	BiasAdd_1g
Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      ÿÿÿÿ2	
Const_1q
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
split_1/split_dimÓ
split_1SplitVBiasAdd_1:output:0Const_1:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2	
split_1g
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Sigmoidk
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	Sigmoid_1d
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mulb
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add_2Q
ReluRelu	add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu\
mul_1MulSigmoid:y:0states*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
sub/x`
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
subd
mul_2Mulsub:z:0Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mul_2_
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add_3 
IdentityIdentity	add_3:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity¤

Identity_1Identity	add_3:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*E
_input_shapes4
2:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ:::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namestates
Ë
¥
while_cond_10216
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_10216___redundant_placeholder03
/while_while_cond_10216___redundant_placeholder13
/while_while_cond_10216___redundant_placeholder23
/while_while_cond_10216___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
:
Ë
¥
while_cond_10923
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_10923___redundant_placeholder03
/while_while_cond_10923___redundant_placeholder13
/while_while_cond_10923___redundant_placeholder23
/while_while_cond_10923___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
Ü	
¬
*__inference_gru_cell_1_layer_call_fn_11457

inputs
states_0
unknown
	unknown_0
	unknown_1
identity

identity_1¢StatefulPartitionedCall£
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_gru_cell_1_layer_call_and_return_conditional_losses_77242
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*E
_input_shapes4
2:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ:::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/0
Ç
¬
E__inference_gru_cell_1_layer_call_and_return_conditional_losses_11389

inputs
states_0
readvariableop_resource"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource
identity

identity_1¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp¢ReadVariableOpx
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:0*
dtype02
ReadVariableOpj
unstackUnpackReadVariableOp:value:0*
T0* 
_output_shapes
:0:0*	
num2	
unstack
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: 0*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02
MatMuls
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constm
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
split/split_dim¬
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
split
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:0*
dtype02
MatMul_1/ReadVariableOp{
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02

MatMul_1y
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02
	BiasAdd_1g
Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      ÿÿÿÿ2	
Const_1q
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
split_1/split_dimÓ
split_1SplitVBiasAdd_1:output:0Const_1:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2	
split_1g
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Sigmoidk
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	Sigmoid_1d
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mulb
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add_2Q
ReluRelu	add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu^
mul_1MulSigmoid:y:0states_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
sub/x`
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
subd
mul_2Mulsub:z:0Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mul_2_
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add_3 
IdentityIdentity	add_3:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity¤

Identity_1Identity	add_3:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*E
_input_shapes4
2:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ:::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/0

E
)__inference_dropout_1_layer_call_fn_11222

inputs
identityÄ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dropout_1_layer_call_and_return_conditional_losses_89192
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
­

#__inference_gru_layer_call_fn_10148
inputs_0
unknown
	unknown_0
	unknown_1
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *F
fAR?
=__inference_gru_layer_call_and_return_conditional_losses_76032
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
üD

while_body_10376
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0,
(while_gru_cell_readvariableop_resource_03
/while_gru_cell_matmul_readvariableop_resource_05
1while_gru_cell_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor*
&while_gru_cell_readvariableop_resource1
-while_gru_cell_matmul_readvariableop_resource3
/while_gru_cell_matmul_1_readvariableop_resource¢$while/gru_cell/MatMul/ReadVariableOp¢&while/gru_cell/MatMul_1/ReadVariableOp¢while/gru_cell/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÓ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem§
while/gru_cell/ReadVariableOpReadVariableOp(while_gru_cell_readvariableop_resource_0*
_output_shapes

:`*
dtype02
while/gru_cell/ReadVariableOp
while/gru_cell/unstackUnpack%while/gru_cell/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num2
while/gru_cell/unstack¼
$while/gru_cell/MatMul/ReadVariableOpReadVariableOp/while_gru_cell_matmul_readvariableop_resource_0*
_output_shapes

:`*
dtype02&
$while/gru_cell/MatMul/ReadVariableOpÊ
while/gru_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0,while/gru_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
while/gru_cell/MatMul¯
while/gru_cell/BiasAddBiasAddwhile/gru_cell/MatMul:product:0while/gru_cell/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
while/gru_cell/BiasAddn
while/gru_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/gru_cell/Const
while/gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2 
while/gru_cell/split/split_dimè
while/gru_cell/splitSplit'while/gru_cell/split/split_dim:output:0while/gru_cell/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
while/gru_cell/splitÂ
&while/gru_cell/MatMul_1/ReadVariableOpReadVariableOp1while_gru_cell_matmul_1_readvariableop_resource_0*
_output_shapes

: `*
dtype02(
&while/gru_cell/MatMul_1/ReadVariableOp³
while/gru_cell/MatMul_1MatMulwhile_placeholder_2.while/gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
while/gru_cell/MatMul_1µ
while/gru_cell/BiasAdd_1BiasAdd!while/gru_cell/MatMul_1:product:0while/gru_cell/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
while/gru_cell/BiasAdd_1
while/gru_cell/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"        ÿÿÿÿ2
while/gru_cell/Const_1
 while/gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2"
 while/gru_cell/split_1/split_dim
while/gru_cell/split_1SplitV!while/gru_cell/BiasAdd_1:output:0while/gru_cell/Const_1:output:0)while/gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
while/gru_cell/split_1£
while/gru_cell/addAddV2while/gru_cell/split:output:0while/gru_cell/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/add
while/gru_cell/SigmoidSigmoidwhile/gru_cell/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/Sigmoid§
while/gru_cell/add_1AddV2while/gru_cell/split:output:1while/gru_cell/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/add_1
while/gru_cell/Sigmoid_1Sigmoidwhile/gru_cell/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/Sigmoid_1 
while/gru_cell/mulMulwhile/gru_cell/Sigmoid_1:y:0while/gru_cell/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/mul
while/gru_cell/add_2AddV2while/gru_cell/split:output:2while/gru_cell/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/add_2~
while/gru_cell/ReluReluwhile/gru_cell/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/Relu
while/gru_cell/mul_1Mulwhile/gru_cell/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/mul_1q
while/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
while/gru_cell/sub/x
while/gru_cell/subSubwhile/gru_cell/sub/x:output:0while/gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/sub 
while/gru_cell/mul_2Mulwhile/gru_cell/sub:z:0!while/gru_cell/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/mul_2
while/gru_cell/add_3AddV2while/gru_cell/mul_1:z:0while/gru_cell/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/add_3Ü
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell/add_3:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1Î
while/IdentityIdentitywhile/add_1:z:0%^while/gru_cell/MatMul/ReadVariableOp'^while/gru_cell/MatMul_1/ReadVariableOp^while/gru_cell/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identityá
while/Identity_1Identitywhile_while_maximum_iterations%^while/gru_cell/MatMul/ReadVariableOp'^while/gru_cell/MatMul_1/ReadVariableOp^while/gru_cell/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1Ð
while/Identity_2Identitywhile/add:z:0%^while/gru_cell/MatMul/ReadVariableOp'^while/gru_cell/MatMul_1/ReadVariableOp^while/gru_cell/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2ý
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0%^while/gru_cell/MatMul/ReadVariableOp'^while/gru_cell/MatMul_1/ReadVariableOp^while/gru_cell/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3ì
while/Identity_4Identitywhile/gru_cell/add_3:z:0%^while/gru_cell/MatMul/ReadVariableOp'^while/gru_cell/MatMul_1/ReadVariableOp^while/gru_cell/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4"d
/while_gru_cell_matmul_1_readvariableop_resource1while_gru_cell_matmul_1_readvariableop_resource_0"`
-while_gru_cell_matmul_readvariableop_resource/while_gru_cell_matmul_readvariableop_resource_0"R
&while_gru_cell_readvariableop_resource(while_gru_cell_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :ÿÿÿÿÿÿÿÿÿ : : :::2L
$while/gru_cell/MatMul/ReadVariableOp$while/gru_cell/MatMul/ReadVariableOp2P
&while/gru_cell/MatMul_1/ReadVariableOp&while/gru_cell/MatMul_1/ReadVariableOp2>
while/gru_cell/ReadVariableOpwhile/gru_cell/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
: 
¦Z
Ò
?__inference_gru_1_layer_call_and_return_conditional_losses_8872

inputs&
"gru_cell_1_readvariableop_resource-
)gru_cell_1_matmul_readvariableop_resource/
+gru_cell_1_matmul_1_readvariableop_resource
identity¢ gru_cell_1/MatMul/ReadVariableOp¢"gru_cell_1/MatMul_1/ReadVariableOp¢gru_cell_1/ReadVariableOp¢whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ü
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2
strided_slice_2
gru_cell_1/ReadVariableOpReadVariableOp"gru_cell_1_readvariableop_resource*
_output_shapes

:0*
dtype02
gru_cell_1/ReadVariableOp
gru_cell_1/unstackUnpack!gru_cell_1/ReadVariableOp:value:0*
T0* 
_output_shapes
:0:0*	
num2
gru_cell_1/unstack®
 gru_cell_1/MatMul/ReadVariableOpReadVariableOp)gru_cell_1_matmul_readvariableop_resource*
_output_shapes

: 0*
dtype02"
 gru_cell_1/MatMul/ReadVariableOp¦
gru_cell_1/MatMulMatMulstrided_slice_2:output:0(gru_cell_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02
gru_cell_1/MatMul
gru_cell_1/BiasAddBiasAddgru_cell_1/MatMul:product:0gru_cell_1/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02
gru_cell_1/BiasAddf
gru_cell_1/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru_cell_1/Const
gru_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
gru_cell_1/split/split_dimØ
gru_cell_1/splitSplit#gru_cell_1/split/split_dim:output:0gru_cell_1/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
gru_cell_1/split´
"gru_cell_1/MatMul_1/ReadVariableOpReadVariableOp+gru_cell_1_matmul_1_readvariableop_resource*
_output_shapes

:0*
dtype02$
"gru_cell_1/MatMul_1/ReadVariableOp¢
gru_cell_1/MatMul_1MatMulzeros:output:0*gru_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02
gru_cell_1/MatMul_1¥
gru_cell_1/BiasAdd_1BiasAddgru_cell_1/MatMul_1:product:0gru_cell_1/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02
gru_cell_1/BiasAdd_1}
gru_cell_1/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      ÿÿÿÿ2
gru_cell_1/Const_1
gru_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
gru_cell_1/split_1/split_dim
gru_cell_1/split_1SplitVgru_cell_1/BiasAdd_1:output:0gru_cell_1/Const_1:output:0%gru_cell_1/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
gru_cell_1/split_1
gru_cell_1/addAddV2gru_cell_1/split:output:0gru_cell_1/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_cell_1/addy
gru_cell_1/SigmoidSigmoidgru_cell_1/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_cell_1/Sigmoid
gru_cell_1/add_1AddV2gru_cell_1/split:output:1gru_cell_1/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_cell_1/add_1
gru_cell_1/Sigmoid_1Sigmoidgru_cell_1/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_cell_1/Sigmoid_1
gru_cell_1/mulMulgru_cell_1/Sigmoid_1:y:0gru_cell_1/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_cell_1/mul
gru_cell_1/add_2AddV2gru_cell_1/split:output:2gru_cell_1/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_cell_1/add_2r
gru_cell_1/ReluRelugru_cell_1/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_cell_1/Relu
gru_cell_1/mul_1Mulgru_cell_1/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_cell_1/mul_1i
gru_cell_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru_cell_1/sub/x
gru_cell_1/subSubgru_cell_1/sub/x:output:0gru_cell_1/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_cell_1/sub
gru_cell_1/mul_2Mulgru_cell_1/sub:z:0gru_cell_1/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_cell_1/mul_2
gru_cell_1/add_3AddV2gru_cell_1/mul_1:z:0gru_cell_1/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_cell_1/add_3
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
TensorArrayV2_1/element_shape¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter£
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0"gru_cell_1_readvariableop_resource)gru_cell_1_matmul_readvariableop_resource+gru_cell_1_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_8782*
condR
while_cond_8781*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   22
0TensorArrayV2Stack/TensorListStack/element_shapeè
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¥
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeØ
IdentityIdentitystrided_slice_3:output:0!^gru_cell_1/MatMul/ReadVariableOp#^gru_cell_1/MatMul_1/ReadVariableOp^gru_cell_1/ReadVariableOp^while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ :::2D
 gru_cell_1/MatMul/ReadVariableOp gru_cell_1/MatMul/ReadVariableOp2H
"gru_cell_1/MatMul_1/ReadVariableOp"gru_cell_1/MatMul_1/ReadVariableOp26
gru_cell_1/ReadVariableOpgru_cell_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
¼
§
B__inference_gru_cell_layer_call_and_return_conditional_losses_7162

inputs

states
readvariableop_resource"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource
identity

identity_1¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp¢ReadVariableOpx
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:`*
dtype02
ReadVariableOpj
unstackUnpackReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num2	
unstack
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:`*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
MatMuls
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constm
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
split/split_dim¬
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
split
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

: `*
dtype02
MatMul_1/ReadVariableOpy
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2

MatMul_1y
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
	BiasAdd_1g
Const_1Const*
_output_shapes
:*
dtype0*!
valueB"        ÿÿÿÿ2	
Const_1q
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
split_1/split_dimÓ
split_1SplitVBiasAdd_1:output:0Const_1:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2	
split_1g
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2	
Sigmoidk
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
	Sigmoid_1d
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
mulb
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
add_2Q
ReluRelu	add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Relu\
mul_1MulSigmoid:y:0states*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
sub/x`
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
subd
mul_2Mulsub:z:0Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
mul_2_
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
add_3 
IdentityIdentity	add_3:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity¤

Identity_1Identity	add_3:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*E
_input_shapes4
2:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ :::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_namestates
Ø	
ª
(__inference_gru_cell_layer_call_fn_11335

inputs
states_0
unknown
	unknown_0
	unknown_1
identity

identity_1¢StatefulPartitionedCall¡
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_gru_cell_layer_call_and_return_conditional_losses_71222
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*E
_input_shapes4
2:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ :::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
states/0
 <
À
=__inference_gru_layer_call_and_return_conditional_losses_7485

inputs
gru_cell_7409
gru_cell_7411
gru_cell_7413
identity¢ gru_cell/StatefulPartitionedCall¢whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ü
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_2Ù
 gru_cell/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0gru_cell_7409gru_cell_7411gru_cell_7413*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_gru_cell_layer_call_and_return_conditional_losses_71222"
 gru_cell/StatefulPartitionedCall
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2
TensorArrayV2_1/element_shape¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterÔ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0gru_cell_7409gru_cell_7411gru_cell_7413*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ : : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_7421*
condR
while_cond_7420*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ : : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    22
0TensorArrayV2Stack/TensorListStack/element_shapeñ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm®
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime
IdentityIdentitytranspose_1:y:0!^gru_cell/StatefulPartitionedCall^while*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::2D
 gru_cell/StatefulPartitionedCall gru_cell/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ðß

D__inference_sequential_layer_call_and_return_conditional_losses_9444

inputs(
$gru_gru_cell_readvariableop_resource/
+gru_gru_cell_matmul_readvariableop_resource1
-gru_gru_cell_matmul_1_readvariableop_resource,
(gru_1_gru_cell_1_readvariableop_resource3
/gru_1_gru_cell_1_matmul_readvariableop_resource5
1gru_1_gru_cell_1_matmul_1_readvariableop_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource
identity¢dense/BiasAdd/ReadVariableOp¢dense/MatMul/ReadVariableOp¢"gru/gru_cell/MatMul/ReadVariableOp¢$gru/gru_cell/MatMul_1/ReadVariableOp¢gru/gru_cell/ReadVariableOp¢	gru/while¢&gru_1/gru_cell_1/MatMul/ReadVariableOp¢(gru_1/gru_cell_1/MatMul_1/ReadVariableOp¢gru_1/gru_cell_1/ReadVariableOp¢gru_1/whileL
	gru/ShapeShapeinputs*
T0*
_output_shapes
:2
	gru/Shape|
gru/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru/strided_slice/stack
gru/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
gru/strided_slice/stack_1
gru/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru/strided_slice/stack_2ú
gru/strided_sliceStridedSlicegru/Shape:output:0 gru/strided_slice/stack:output:0"gru/strided_slice/stack_1:output:0"gru/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
gru/strided_sliced
gru/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
gru/zeros/mul/y|
gru/zeros/mulMulgru/strided_slice:output:0gru/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
gru/zeros/mulg
gru/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
gru/zeros/Less/yw
gru/zeros/LessLessgru/zeros/mul:z:0gru/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
gru/zeros/Lessj
gru/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
gru/zeros/packed/1
gru/zeros/packedPackgru/strided_slice:output:0gru/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
gru/zeros/packedg
gru/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
gru/zeros/Const
	gru/zerosFillgru/zeros/packed:output:0gru/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
	gru/zeros}
gru/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
gru/transpose/perm
gru/transpose	Transposeinputsgru/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru/transpose[
gru/Shape_1Shapegru/transpose:y:0*
T0*
_output_shapes
:2
gru/Shape_1
gru/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru/strided_slice_1/stack
gru/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
gru/strided_slice_1/stack_1
gru/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru/strided_slice_1/stack_2
gru/strided_slice_1StridedSlicegru/Shape_1:output:0"gru/strided_slice_1/stack:output:0$gru/strided_slice_1/stack_1:output:0$gru/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
gru/strided_slice_1
gru/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2!
gru/TensorArrayV2/element_shapeÂ
gru/TensorArrayV2TensorListReserve(gru/TensorArrayV2/element_shape:output:0gru/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
gru/TensorArrayV2Ç
9gru/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2;
9gru/TensorArrayUnstack/TensorListFromTensor/element_shape
+gru/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorgru/transpose:y:0Bgru/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02-
+gru/TensorArrayUnstack/TensorListFromTensor
gru/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru/strided_slice_2/stack
gru/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
gru/strided_slice_2/stack_1
gru/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru/strided_slice_2/stack_2
gru/strided_slice_2StridedSlicegru/transpose:y:0"gru/strided_slice_2/stack:output:0$gru/strided_slice_2/stack_1:output:0$gru/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
gru/strided_slice_2
gru/gru_cell/ReadVariableOpReadVariableOp$gru_gru_cell_readvariableop_resource*
_output_shapes

:`*
dtype02
gru/gru_cell/ReadVariableOp
gru/gru_cell/unstackUnpack#gru/gru_cell/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num2
gru/gru_cell/unstack´
"gru/gru_cell/MatMul/ReadVariableOpReadVariableOp+gru_gru_cell_matmul_readvariableop_resource*
_output_shapes

:`*
dtype02$
"gru/gru_cell/MatMul/ReadVariableOp°
gru/gru_cell/MatMulMatMulgru/strided_slice_2:output:0*gru/gru_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
gru/gru_cell/MatMul§
gru/gru_cell/BiasAddBiasAddgru/gru_cell/MatMul:product:0gru/gru_cell/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
gru/gru_cell/BiasAddj
gru/gru_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru/gru_cell/Const
gru/gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
gru/gru_cell/split/split_dimà
gru/gru_cell/splitSplit%gru/gru_cell/split/split_dim:output:0gru/gru_cell/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
gru/gru_cell/splitº
$gru/gru_cell/MatMul_1/ReadVariableOpReadVariableOp-gru_gru_cell_matmul_1_readvariableop_resource*
_output_shapes

: `*
dtype02&
$gru/gru_cell/MatMul_1/ReadVariableOp¬
gru/gru_cell/MatMul_1MatMulgru/zeros:output:0,gru/gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
gru/gru_cell/MatMul_1­
gru/gru_cell/BiasAdd_1BiasAddgru/gru_cell/MatMul_1:product:0gru/gru_cell/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
gru/gru_cell/BiasAdd_1
gru/gru_cell/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"        ÿÿÿÿ2
gru/gru_cell/Const_1
gru/gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2 
gru/gru_cell/split_1/split_dim
gru/gru_cell/split_1SplitVgru/gru_cell/BiasAdd_1:output:0gru/gru_cell/Const_1:output:0'gru/gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
gru/gru_cell/split_1
gru/gru_cell/addAddV2gru/gru_cell/split:output:0gru/gru_cell/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/gru_cell/add
gru/gru_cell/SigmoidSigmoidgru/gru_cell/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/gru_cell/Sigmoid
gru/gru_cell/add_1AddV2gru/gru_cell/split:output:1gru/gru_cell/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/gru_cell/add_1
gru/gru_cell/Sigmoid_1Sigmoidgru/gru_cell/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/gru_cell/Sigmoid_1
gru/gru_cell/mulMulgru/gru_cell/Sigmoid_1:y:0gru/gru_cell/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/gru_cell/mul
gru/gru_cell/add_2AddV2gru/gru_cell/split:output:2gru/gru_cell/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/gru_cell/add_2x
gru/gru_cell/ReluRelugru/gru_cell/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/gru_cell/Relu
gru/gru_cell/mul_1Mulgru/gru_cell/Sigmoid:y:0gru/zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/gru_cell/mul_1m
gru/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru/gru_cell/sub/x
gru/gru_cell/subSubgru/gru_cell/sub/x:output:0gru/gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/gru_cell/sub
gru/gru_cell/mul_2Mulgru/gru_cell/sub:z:0gru/gru_cell/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/gru_cell/mul_2
gru/gru_cell/add_3AddV2gru/gru_cell/mul_1:z:0gru/gru_cell/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/gru_cell/add_3
!gru/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2#
!gru/TensorArrayV2_1/element_shapeÈ
gru/TensorArrayV2_1TensorListReserve*gru/TensorArrayV2_1/element_shape:output:0gru/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
gru/TensorArrayV2_1V
gru/timeConst*
_output_shapes
: *
dtype0*
value	B : 2

gru/time
gru/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
gru/while/maximum_iterationsr
gru/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
gru/while/loop_counterÕ
	gru/whileWhilegru/while/loop_counter:output:0%gru/while/maximum_iterations:output:0gru/time:output:0gru/TensorArrayV2_1:handle:0gru/zeros:output:0gru/strided_slice_1:output:0;gru/TensorArrayUnstack/TensorListFromTensor:output_handle:0$gru_gru_cell_readvariableop_resource+gru_gru_cell_matmul_readvariableop_resource-gru_gru_cell_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ : : : : : *%
_read_only_resource_inputs
	*
bodyR
gru_while_body_9177*
condR
gru_while_cond_9176*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ : : : : : *
parallel_iterations 2
	gru/while½
4gru/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    26
4gru/TensorArrayV2Stack/TensorListStack/element_shapeø
&gru/TensorArrayV2Stack/TensorListStackTensorListStackgru/while:output:3=gru/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype02(
&gru/TensorArrayV2Stack/TensorListStack
gru/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
gru/strided_slice_3/stack
gru/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
gru/strided_slice_3/stack_1
gru/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru/strided_slice_3/stack_2²
gru/strided_slice_3StridedSlice/gru/TensorArrayV2Stack/TensorListStack:tensor:0"gru/strided_slice_3/stack:output:0$gru/strided_slice_3/stack_1:output:0$gru/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2
gru/strided_slice_3
gru/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
gru/transpose_1/permµ
gru/transpose_1	Transpose/gru/TensorArrayV2Stack/TensorListStack:tensor:0gru/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/transpose_1n
gru/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
gru/runtimes
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
dropout/dropout/Const
dropout/dropout/MulMulgru/transpose_1:y:0dropout/dropout/Const:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dropout/dropout/Mulq
dropout/dropout/ShapeShapegru/transpose_1:y:0*
T0*
_output_shapes
:2
dropout/dropout/ShapeÐ
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
dtype02.
,dropout/dropout/random_uniform/RandomUniform
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dropout/dropout/GreaterEqual/yâ
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dropout/dropout/GreaterEqual
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dropout/dropout/Cast
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dropout/dropout/Mul_1c
gru_1/ShapeShapedropout/dropout/Mul_1:z:0*
T0*
_output_shapes
:2
gru_1/Shape
gru_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru_1/strided_slice/stack
gru_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
gru_1/strided_slice/stack_1
gru_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru_1/strided_slice/stack_2
gru_1/strided_sliceStridedSlicegru_1/Shape:output:0"gru_1/strided_slice/stack:output:0$gru_1/strided_slice/stack_1:output:0$gru_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
gru_1/strided_sliceh
gru_1/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
gru_1/zeros/mul/y
gru_1/zeros/mulMulgru_1/strided_slice:output:0gru_1/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
gru_1/zeros/mulk
gru_1/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
gru_1/zeros/Less/y
gru_1/zeros/LessLessgru_1/zeros/mul:z:0gru_1/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
gru_1/zeros/Lessn
gru_1/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
gru_1/zeros/packed/1
gru_1/zeros/packedPackgru_1/strided_slice:output:0gru_1/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
gru_1/zeros/packedk
gru_1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
gru_1/zeros/Const
gru_1/zerosFillgru_1/zeros/packed:output:0gru_1/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_1/zeros
gru_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
gru_1/transpose/perm
gru_1/transpose	Transposedropout/dropout/Mul_1:z:0gru_1/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_1/transposea
gru_1/Shape_1Shapegru_1/transpose:y:0*
T0*
_output_shapes
:2
gru_1/Shape_1
gru_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru_1/strided_slice_1/stack
gru_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
gru_1/strided_slice_1/stack_1
gru_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru_1/strided_slice_1/stack_2
gru_1/strided_slice_1StridedSlicegru_1/Shape_1:output:0$gru_1/strided_slice_1/stack:output:0&gru_1/strided_slice_1/stack_1:output:0&gru_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
gru_1/strided_slice_1
!gru_1/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2#
!gru_1/TensorArrayV2/element_shapeÊ
gru_1/TensorArrayV2TensorListReserve*gru_1/TensorArrayV2/element_shape:output:0gru_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
gru_1/TensorArrayV2Ë
;gru_1/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2=
;gru_1/TensorArrayUnstack/TensorListFromTensor/element_shape
-gru_1/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorgru_1/transpose:y:0Dgru_1/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02/
-gru_1/TensorArrayUnstack/TensorListFromTensor
gru_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru_1/strided_slice_2/stack
gru_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
gru_1/strided_slice_2/stack_1
gru_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru_1/strided_slice_2/stack_2 
gru_1/strided_slice_2StridedSlicegru_1/transpose:y:0$gru_1/strided_slice_2/stack:output:0&gru_1/strided_slice_2/stack_1:output:0&gru_1/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2
gru_1/strided_slice_2«
gru_1/gru_cell_1/ReadVariableOpReadVariableOp(gru_1_gru_cell_1_readvariableop_resource*
_output_shapes

:0*
dtype02!
gru_1/gru_cell_1/ReadVariableOp
gru_1/gru_cell_1/unstackUnpack'gru_1/gru_cell_1/ReadVariableOp:value:0*
T0* 
_output_shapes
:0:0*	
num2
gru_1/gru_cell_1/unstackÀ
&gru_1/gru_cell_1/MatMul/ReadVariableOpReadVariableOp/gru_1_gru_cell_1_matmul_readvariableop_resource*
_output_shapes

: 0*
dtype02(
&gru_1/gru_cell_1/MatMul/ReadVariableOp¾
gru_1/gru_cell_1/MatMulMatMulgru_1/strided_slice_2:output:0.gru_1/gru_cell_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02
gru_1/gru_cell_1/MatMul·
gru_1/gru_cell_1/BiasAddBiasAdd!gru_1/gru_cell_1/MatMul:product:0!gru_1/gru_cell_1/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02
gru_1/gru_cell_1/BiasAddr
gru_1/gru_cell_1/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru_1/gru_cell_1/Const
 gru_1/gru_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2"
 gru_1/gru_cell_1/split/split_dimð
gru_1/gru_cell_1/splitSplit)gru_1/gru_cell_1/split/split_dim:output:0!gru_1/gru_cell_1/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
gru_1/gru_cell_1/splitÆ
(gru_1/gru_cell_1/MatMul_1/ReadVariableOpReadVariableOp1gru_1_gru_cell_1_matmul_1_readvariableop_resource*
_output_shapes

:0*
dtype02*
(gru_1/gru_cell_1/MatMul_1/ReadVariableOpº
gru_1/gru_cell_1/MatMul_1MatMulgru_1/zeros:output:00gru_1/gru_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02
gru_1/gru_cell_1/MatMul_1½
gru_1/gru_cell_1/BiasAdd_1BiasAdd#gru_1/gru_cell_1/MatMul_1:product:0!gru_1/gru_cell_1/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02
gru_1/gru_cell_1/BiasAdd_1
gru_1/gru_cell_1/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      ÿÿÿÿ2
gru_1/gru_cell_1/Const_1
"gru_1/gru_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2$
"gru_1/gru_cell_1/split_1/split_dim¨
gru_1/gru_cell_1/split_1SplitV#gru_1/gru_cell_1/BiasAdd_1:output:0!gru_1/gru_cell_1/Const_1:output:0+gru_1/gru_cell_1/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
gru_1/gru_cell_1/split_1«
gru_1/gru_cell_1/addAddV2gru_1/gru_cell_1/split:output:0!gru_1/gru_cell_1/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_1/gru_cell_1/add
gru_1/gru_cell_1/SigmoidSigmoidgru_1/gru_cell_1/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_1/gru_cell_1/Sigmoid¯
gru_1/gru_cell_1/add_1AddV2gru_1/gru_cell_1/split:output:1!gru_1/gru_cell_1/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_1/gru_cell_1/add_1
gru_1/gru_cell_1/Sigmoid_1Sigmoidgru_1/gru_cell_1/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_1/gru_cell_1/Sigmoid_1¨
gru_1/gru_cell_1/mulMulgru_1/gru_cell_1/Sigmoid_1:y:0!gru_1/gru_cell_1/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_1/gru_cell_1/mul¦
gru_1/gru_cell_1/add_2AddV2gru_1/gru_cell_1/split:output:2gru_1/gru_cell_1/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_1/gru_cell_1/add_2
gru_1/gru_cell_1/ReluRelugru_1/gru_cell_1/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_1/gru_cell_1/Relu
gru_1/gru_cell_1/mul_1Mulgru_1/gru_cell_1/Sigmoid:y:0gru_1/zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_1/gru_cell_1/mul_1u
gru_1/gru_cell_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru_1/gru_cell_1/sub/x¤
gru_1/gru_cell_1/subSubgru_1/gru_cell_1/sub/x:output:0gru_1/gru_cell_1/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_1/gru_cell_1/sub¨
gru_1/gru_cell_1/mul_2Mulgru_1/gru_cell_1/sub:z:0#gru_1/gru_cell_1/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_1/gru_cell_1/mul_2£
gru_1/gru_cell_1/add_3AddV2gru_1/gru_cell_1/mul_1:z:0gru_1/gru_cell_1/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_1/gru_cell_1/add_3
#gru_1/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2%
#gru_1/TensorArrayV2_1/element_shapeÐ
gru_1/TensorArrayV2_1TensorListReserve,gru_1/TensorArrayV2_1/element_shape:output:0gru_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
gru_1/TensorArrayV2_1Z

gru_1/timeConst*
_output_shapes
: *
dtype0*
value	B : 2

gru_1/time
gru_1/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2 
gru_1/while/maximum_iterationsv
gru_1/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
gru_1/while/loop_counter÷
gru_1/whileWhile!gru_1/while/loop_counter:output:0'gru_1/while/maximum_iterations:output:0gru_1/time:output:0gru_1/TensorArrayV2_1:handle:0gru_1/zeros:output:0gru_1/strided_slice_1:output:0=gru_1/TensorArrayUnstack/TensorListFromTensor:output_handle:0(gru_1_gru_cell_1_readvariableop_resource/gru_1_gru_cell_1_matmul_readvariableop_resource1gru_1_gru_cell_1_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	*!
bodyR
gru_1_while_body_9340*!
condR
gru_1_while_cond_9339*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 2
gru_1/whileÁ
6gru_1/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   28
6gru_1/TensorArrayV2Stack/TensorListStack/element_shape
(gru_1/TensorArrayV2Stack/TensorListStackTensorListStackgru_1/while:output:3?gru_1/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02*
(gru_1/TensorArrayV2Stack/TensorListStack
gru_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
gru_1/strided_slice_3/stack
gru_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
gru_1/strided_slice_3/stack_1
gru_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru_1/strided_slice_3/stack_2¾
gru_1/strided_slice_3StridedSlice1gru_1/TensorArrayV2Stack/TensorListStack:tensor:0$gru_1/strided_slice_3/stack:output:0&gru_1/strided_slice_3/stack_1:output:0&gru_1/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
gru_1/strided_slice_3
gru_1/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
gru_1/transpose_1/perm½
gru_1/transpose_1	Transpose1gru_1/TensorArrayV2Stack/TensorListStack:tensor:0gru_1/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_1/transpose_1r
gru_1/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
gru_1/runtimew
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
dropout_1/dropout/Const©
dropout_1/dropout/MulMulgru_1/strided_slice_3:output:0 dropout_1/dropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout_1/dropout/Mul
dropout_1/dropout/ShapeShapegru_1/strided_slice_3:output:0*
T0*
_output_shapes
:2
dropout_1/dropout/ShapeÒ
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype020
.dropout_1/dropout/random_uniform/RandomUniform
 dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dropout_1/dropout/GreaterEqual/yæ
dropout_1/dropout/GreaterEqualGreaterEqual7dropout_1/dropout/random_uniform/RandomUniform:output:0)dropout_1/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
dropout_1/dropout/GreaterEqual
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout_1/dropout/Cast¢
dropout_1/dropout/Mul_1Muldropout_1/dropout/Mul:z:0dropout_1/dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout_1/dropout/Mul_1
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense/MatMul/ReadVariableOp
dense/MatMulMatMuldropout_1/dropout/Mul_1:z:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/MatMul
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
dense/BiasAdd/ReadVariableOp
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/BiasAdd¡
IdentityIdentitydense/BiasAdd:output:0^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp#^gru/gru_cell/MatMul/ReadVariableOp%^gru/gru_cell/MatMul_1/ReadVariableOp^gru/gru_cell/ReadVariableOp
^gru/while'^gru_1/gru_cell_1/MatMul/ReadVariableOp)^gru_1/gru_cell_1/MatMul_1/ReadVariableOp ^gru_1/gru_cell_1/ReadVariableOp^gru_1/while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:ÿÿÿÿÿÿÿÿÿ::::::::2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2H
"gru/gru_cell/MatMul/ReadVariableOp"gru/gru_cell/MatMul/ReadVariableOp2L
$gru/gru_cell/MatMul_1/ReadVariableOp$gru/gru_cell/MatMul_1/ReadVariableOp2:
gru/gru_cell/ReadVariableOpgru/gru_cell/ReadVariableOp2
	gru/while	gru/while2P
&gru_1/gru_cell_1/MatMul/ReadVariableOp&gru_1/gru_cell_1/MatMul/ReadVariableOp2T
(gru_1/gru_cell_1/MatMul_1/ReadVariableOp(gru_1/gru_cell_1/MatMul_1/ReadVariableOp2B
gru_1/gru_cell_1/ReadVariableOpgru_1/gru_cell_1/ReadVariableOp2
gru_1/whilegru_1/while:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


#__inference_gru_layer_call_fn_10477

inputs
unknown
	unknown_0
	unknown_1
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *F
fAR?
=__inference_gru_layer_call_and_return_conditional_losses_83362
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ:::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Æ
a
C__inference_dropout_1_layer_call_and_return_conditional_losses_8919

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ÿ

b
C__inference_dropout_1_layer_call_and_return_conditional_losses_8914

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape´
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dropout/GreaterEqual/y¾
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¾<
Ê
?__inference_gru_1_layer_call_and_return_conditional_losses_8047

inputs
gru_cell_1_7971
gru_cell_1_7973
gru_cell_1_7975
identity¢"gru_cell_1/StatefulPartitionedCall¢whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ü
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2
strided_slice_2å
"gru_cell_1/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0gru_cell_1_7971gru_cell_1_7973gru_cell_1_7975*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_gru_cell_1_layer_call_and_return_conditional_losses_76842$
"gru_cell_1/StatefulPartitionedCall
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
TensorArrayV2_1/element_shape¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterÚ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0gru_cell_1_7971gru_cell_1_7973gru_cell_1_7975*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_7983*
condR
while_cond_7982*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   22
0TensorArrayV2Stack/TensorListStack/element_shapeñ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm®
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime
IdentityIdentitystrided_slice_3:output:0#^gru_cell_1/StatefulPartitionedCall^while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ :::2H
"gru_cell_1/StatefulPartitionedCall"gru_cell_1/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ÿ

%__inference_gru_1_layer_call_fn_10855

inputs
unknown
	unknown_0
	unknown_1
identity¢StatefulPartitionedCallÿ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_gru_1_layer_call_and_return_conditional_losses_88722
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ :::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ÃP
×
gru_1_while_body_9340(
$gru_1_while_gru_1_while_loop_counter.
*gru_1_while_gru_1_while_maximum_iterations
gru_1_while_placeholder
gru_1_while_placeholder_1
gru_1_while_placeholder_2'
#gru_1_while_gru_1_strided_slice_1_0c
_gru_1_while_tensorarrayv2read_tensorlistgetitem_gru_1_tensorarrayunstack_tensorlistfromtensor_04
0gru_1_while_gru_cell_1_readvariableop_resource_0;
7gru_1_while_gru_cell_1_matmul_readvariableop_resource_0=
9gru_1_while_gru_cell_1_matmul_1_readvariableop_resource_0
gru_1_while_identity
gru_1_while_identity_1
gru_1_while_identity_2
gru_1_while_identity_3
gru_1_while_identity_4%
!gru_1_while_gru_1_strided_slice_1a
]gru_1_while_tensorarrayv2read_tensorlistgetitem_gru_1_tensorarrayunstack_tensorlistfromtensor2
.gru_1_while_gru_cell_1_readvariableop_resource9
5gru_1_while_gru_cell_1_matmul_readvariableop_resource;
7gru_1_while_gru_cell_1_matmul_1_readvariableop_resource¢,gru_1/while/gru_cell_1/MatMul/ReadVariableOp¢.gru_1/while/gru_cell_1/MatMul_1/ReadVariableOp¢%gru_1/while/gru_cell_1/ReadVariableOpÏ
=gru_1/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2?
=gru_1/while/TensorArrayV2Read/TensorListGetItem/element_shape÷
/gru_1/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem_gru_1_while_tensorarrayv2read_tensorlistgetitem_gru_1_tensorarrayunstack_tensorlistfromtensor_0gru_1_while_placeholderFgru_1/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype021
/gru_1/while/TensorArrayV2Read/TensorListGetItem¿
%gru_1/while/gru_cell_1/ReadVariableOpReadVariableOp0gru_1_while_gru_cell_1_readvariableop_resource_0*
_output_shapes

:0*
dtype02'
%gru_1/while/gru_cell_1/ReadVariableOp¯
gru_1/while/gru_cell_1/unstackUnpack-gru_1/while/gru_cell_1/ReadVariableOp:value:0*
T0* 
_output_shapes
:0:0*	
num2 
gru_1/while/gru_cell_1/unstackÔ
,gru_1/while/gru_cell_1/MatMul/ReadVariableOpReadVariableOp7gru_1_while_gru_cell_1_matmul_readvariableop_resource_0*
_output_shapes

: 0*
dtype02.
,gru_1/while/gru_cell_1/MatMul/ReadVariableOpè
gru_1/while/gru_cell_1/MatMulMatMul6gru_1/while/TensorArrayV2Read/TensorListGetItem:item:04gru_1/while/gru_cell_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02
gru_1/while/gru_cell_1/MatMulÏ
gru_1/while/gru_cell_1/BiasAddBiasAdd'gru_1/while/gru_cell_1/MatMul:product:0'gru_1/while/gru_cell_1/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02 
gru_1/while/gru_cell_1/BiasAdd~
gru_1/while/gru_cell_1/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru_1/while/gru_cell_1/Const
&gru_1/while/gru_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2(
&gru_1/while/gru_cell_1/split/split_dim
gru_1/while/gru_cell_1/splitSplit/gru_1/while/gru_cell_1/split/split_dim:output:0'gru_1/while/gru_cell_1/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
gru_1/while/gru_cell_1/splitÚ
.gru_1/while/gru_cell_1/MatMul_1/ReadVariableOpReadVariableOp9gru_1_while_gru_cell_1_matmul_1_readvariableop_resource_0*
_output_shapes

:0*
dtype020
.gru_1/while/gru_cell_1/MatMul_1/ReadVariableOpÑ
gru_1/while/gru_cell_1/MatMul_1MatMulgru_1_while_placeholder_26gru_1/while/gru_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02!
gru_1/while/gru_cell_1/MatMul_1Õ
 gru_1/while/gru_cell_1/BiasAdd_1BiasAdd)gru_1/while/gru_cell_1/MatMul_1:product:0'gru_1/while/gru_cell_1/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02"
 gru_1/while/gru_cell_1/BiasAdd_1
gru_1/while/gru_cell_1/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      ÿÿÿÿ2 
gru_1/while/gru_cell_1/Const_1
(gru_1/while/gru_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2*
(gru_1/while/gru_cell_1/split_1/split_dimÆ
gru_1/while/gru_cell_1/split_1SplitV)gru_1/while/gru_cell_1/BiasAdd_1:output:0'gru_1/while/gru_cell_1/Const_1:output:01gru_1/while/gru_cell_1/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2 
gru_1/while/gru_cell_1/split_1Ã
gru_1/while/gru_cell_1/addAddV2%gru_1/while/gru_cell_1/split:output:0'gru_1/while/gru_cell_1/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_1/while/gru_cell_1/add
gru_1/while/gru_cell_1/SigmoidSigmoidgru_1/while/gru_cell_1/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
gru_1/while/gru_cell_1/SigmoidÇ
gru_1/while/gru_cell_1/add_1AddV2%gru_1/while/gru_cell_1/split:output:1'gru_1/while/gru_cell_1/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_1/while/gru_cell_1/add_1£
 gru_1/while/gru_cell_1/Sigmoid_1Sigmoid gru_1/while/gru_cell_1/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 gru_1/while/gru_cell_1/Sigmoid_1À
gru_1/while/gru_cell_1/mulMul$gru_1/while/gru_cell_1/Sigmoid_1:y:0'gru_1/while/gru_cell_1/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_1/while/gru_cell_1/mul¾
gru_1/while/gru_cell_1/add_2AddV2%gru_1/while/gru_cell_1/split:output:2gru_1/while/gru_cell_1/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_1/while/gru_cell_1/add_2
gru_1/while/gru_cell_1/ReluRelu gru_1/while/gru_cell_1/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_1/while/gru_cell_1/Relu´
gru_1/while/gru_cell_1/mul_1Mul"gru_1/while/gru_cell_1/Sigmoid:y:0gru_1_while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_1/while/gru_cell_1/mul_1
gru_1/while/gru_cell_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru_1/while/gru_cell_1/sub/x¼
gru_1/while/gru_cell_1/subSub%gru_1/while/gru_cell_1/sub/x:output:0"gru_1/while/gru_cell_1/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_1/while/gru_cell_1/subÀ
gru_1/while/gru_cell_1/mul_2Mulgru_1/while/gru_cell_1/sub:z:0)gru_1/while/gru_cell_1/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_1/while/gru_cell_1/mul_2»
gru_1/while/gru_cell_1/add_3AddV2 gru_1/while/gru_cell_1/mul_1:z:0 gru_1/while/gru_cell_1/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_1/while/gru_cell_1/add_3ü
0gru_1/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemgru_1_while_placeholder_1gru_1_while_placeholder gru_1/while/gru_cell_1/add_3:z:0*
_output_shapes
: *
element_dtype022
0gru_1/while/TensorArrayV2Write/TensorListSetItemh
gru_1/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
gru_1/while/add/y
gru_1/while/addAddV2gru_1_while_placeholdergru_1/while/add/y:output:0*
T0*
_output_shapes
: 2
gru_1/while/addl
gru_1/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
gru_1/while/add_1/y
gru_1/while/add_1AddV2$gru_1_while_gru_1_while_loop_countergru_1/while/add_1/y:output:0*
T0*
_output_shapes
: 2
gru_1/while/add_1ø
gru_1/while/IdentityIdentitygru_1/while/add_1:z:0-^gru_1/while/gru_cell_1/MatMul/ReadVariableOp/^gru_1/while/gru_cell_1/MatMul_1/ReadVariableOp&^gru_1/while/gru_cell_1/ReadVariableOp*
T0*
_output_shapes
: 2
gru_1/while/Identity
gru_1/while/Identity_1Identity*gru_1_while_gru_1_while_maximum_iterations-^gru_1/while/gru_cell_1/MatMul/ReadVariableOp/^gru_1/while/gru_cell_1/MatMul_1/ReadVariableOp&^gru_1/while/gru_cell_1/ReadVariableOp*
T0*
_output_shapes
: 2
gru_1/while/Identity_1ú
gru_1/while/Identity_2Identitygru_1/while/add:z:0-^gru_1/while/gru_cell_1/MatMul/ReadVariableOp/^gru_1/while/gru_cell_1/MatMul_1/ReadVariableOp&^gru_1/while/gru_cell_1/ReadVariableOp*
T0*
_output_shapes
: 2
gru_1/while/Identity_2§
gru_1/while/Identity_3Identity@gru_1/while/TensorArrayV2Write/TensorListSetItem:output_handle:0-^gru_1/while/gru_cell_1/MatMul/ReadVariableOp/^gru_1/while/gru_cell_1/MatMul_1/ReadVariableOp&^gru_1/while/gru_cell_1/ReadVariableOp*
T0*
_output_shapes
: 2
gru_1/while/Identity_3
gru_1/while/Identity_4Identity gru_1/while/gru_cell_1/add_3:z:0-^gru_1/while/gru_cell_1/MatMul/ReadVariableOp/^gru_1/while/gru_cell_1/MatMul_1/ReadVariableOp&^gru_1/while/gru_cell_1/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_1/while/Identity_4"H
!gru_1_while_gru_1_strided_slice_1#gru_1_while_gru_1_strided_slice_1_0"t
7gru_1_while_gru_cell_1_matmul_1_readvariableop_resource9gru_1_while_gru_cell_1_matmul_1_readvariableop_resource_0"p
5gru_1_while_gru_cell_1_matmul_readvariableop_resource7gru_1_while_gru_cell_1_matmul_readvariableop_resource_0"b
.gru_1_while_gru_cell_1_readvariableop_resource0gru_1_while_gru_cell_1_readvariableop_resource_0"5
gru_1_while_identitygru_1/while/Identity:output:0"9
gru_1_while_identity_1gru_1/while/Identity_1:output:0"9
gru_1_while_identity_2gru_1/while/Identity_2:output:0"9
gru_1_while_identity_3gru_1/while/Identity_3:output:0"9
gru_1_while_identity_4gru_1/while/Identity_4:output:0"À
]gru_1_while_tensorarrayv2read_tensorlistgetitem_gru_1_tensorarrayunstack_tensorlistfromtensor_gru_1_while_tensorarrayv2read_tensorlistgetitem_gru_1_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :ÿÿÿÿÿÿÿÿÿ: : :::2\
,gru_1/while/gru_cell_1/MatMul/ReadVariableOp,gru_1/while/gru_cell_1/MatMul/ReadVariableOp2`
.gru_1/while/gru_cell_1/MatMul_1/ReadVariableOp.gru_1/while/gru_cell_1/MatMul_1/ReadVariableOp2N
%gru_1/while/gru_cell_1/ReadVariableOp%gru_1/while/gru_cell_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
òF
£
while_body_8782
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0.
*while_gru_cell_1_readvariableop_resource_05
1while_gru_cell_1_matmul_readvariableop_resource_07
3while_gru_cell_1_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor,
(while_gru_cell_1_readvariableop_resource3
/while_gru_cell_1_matmul_readvariableop_resource5
1while_gru_cell_1_matmul_1_readvariableop_resource¢&while/gru_cell_1/MatMul/ReadVariableOp¢(while/gru_cell_1/MatMul_1/ReadVariableOp¢while/gru_cell_1/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÓ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem­
while/gru_cell_1/ReadVariableOpReadVariableOp*while_gru_cell_1_readvariableop_resource_0*
_output_shapes

:0*
dtype02!
while/gru_cell_1/ReadVariableOp
while/gru_cell_1/unstackUnpack'while/gru_cell_1/ReadVariableOp:value:0*
T0* 
_output_shapes
:0:0*	
num2
while/gru_cell_1/unstackÂ
&while/gru_cell_1/MatMul/ReadVariableOpReadVariableOp1while_gru_cell_1_matmul_readvariableop_resource_0*
_output_shapes

: 0*
dtype02(
&while/gru_cell_1/MatMul/ReadVariableOpÐ
while/gru_cell_1/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/gru_cell_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02
while/gru_cell_1/MatMul·
while/gru_cell_1/BiasAddBiasAdd!while/gru_cell_1/MatMul:product:0!while/gru_cell_1/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02
while/gru_cell_1/BiasAddr
while/gru_cell_1/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/gru_cell_1/Const
 while/gru_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2"
 while/gru_cell_1/split/split_dimð
while/gru_cell_1/splitSplit)while/gru_cell_1/split/split_dim:output:0!while/gru_cell_1/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
while/gru_cell_1/splitÈ
(while/gru_cell_1/MatMul_1/ReadVariableOpReadVariableOp3while_gru_cell_1_matmul_1_readvariableop_resource_0*
_output_shapes

:0*
dtype02*
(while/gru_cell_1/MatMul_1/ReadVariableOp¹
while/gru_cell_1/MatMul_1MatMulwhile_placeholder_20while/gru_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02
while/gru_cell_1/MatMul_1½
while/gru_cell_1/BiasAdd_1BiasAdd#while/gru_cell_1/MatMul_1:product:0!while/gru_cell_1/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02
while/gru_cell_1/BiasAdd_1
while/gru_cell_1/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      ÿÿÿÿ2
while/gru_cell_1/Const_1
"while/gru_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2$
"while/gru_cell_1/split_1/split_dim¨
while/gru_cell_1/split_1SplitV#while/gru_cell_1/BiasAdd_1:output:0!while/gru_cell_1/Const_1:output:0+while/gru_cell_1/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
while/gru_cell_1/split_1«
while/gru_cell_1/addAddV2while/gru_cell_1/split:output:0!while/gru_cell_1/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_1/add
while/gru_cell_1/SigmoidSigmoidwhile/gru_cell_1/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_1/Sigmoid¯
while/gru_cell_1/add_1AddV2while/gru_cell_1/split:output:1!while/gru_cell_1/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_1/add_1
while/gru_cell_1/Sigmoid_1Sigmoidwhile/gru_cell_1/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_1/Sigmoid_1¨
while/gru_cell_1/mulMulwhile/gru_cell_1/Sigmoid_1:y:0!while/gru_cell_1/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_1/mul¦
while/gru_cell_1/add_2AddV2while/gru_cell_1/split:output:2while/gru_cell_1/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_1/add_2
while/gru_cell_1/ReluReluwhile/gru_cell_1/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_1/Relu
while/gru_cell_1/mul_1Mulwhile/gru_cell_1/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_1/mul_1u
while/gru_cell_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
while/gru_cell_1/sub/x¤
while/gru_cell_1/subSubwhile/gru_cell_1/sub/x:output:0while/gru_cell_1/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_1/sub¨
while/gru_cell_1/mul_2Mulwhile/gru_cell_1/sub:z:0#while/gru_cell_1/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_1/mul_2£
while/gru_cell_1/add_3AddV2while/gru_cell_1/mul_1:z:0while/gru_cell_1/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_1/add_3Þ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_1/add_3:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1Ô
while/IdentityIdentitywhile/add_1:z:0'^while/gru_cell_1/MatMul/ReadVariableOp)^while/gru_cell_1/MatMul_1/ReadVariableOp ^while/gru_cell_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identityç
while/Identity_1Identitywhile_while_maximum_iterations'^while/gru_cell_1/MatMul/ReadVariableOp)^while/gru_cell_1/MatMul_1/ReadVariableOp ^while/gru_cell_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1Ö
while/Identity_2Identitywhile/add:z:0'^while/gru_cell_1/MatMul/ReadVariableOp)^while/gru_cell_1/MatMul_1/ReadVariableOp ^while/gru_cell_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0'^while/gru_cell_1/MatMul/ReadVariableOp)^while/gru_cell_1/MatMul_1/ReadVariableOp ^while/gru_cell_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3ô
while/Identity_4Identitywhile/gru_cell_1/add_3:z:0'^while/gru_cell_1/MatMul/ReadVariableOp)^while/gru_cell_1/MatMul_1/ReadVariableOp ^while/gru_cell_1/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_4"h
1while_gru_cell_1_matmul_1_readvariableop_resource3while_gru_cell_1_matmul_1_readvariableop_resource_0"d
/while_gru_cell_1_matmul_readvariableop_resource1while_gru_cell_1_matmul_readvariableop_resource_0"V
(while_gru_cell_1_readvariableop_resource*while_gru_cell_1_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :ÿÿÿÿÿÿÿÿÿ: : :::2P
&while/gru_cell_1/MatMul/ReadVariableOp&while/gru_cell_1/MatMul/ReadVariableOp2T
(while/gru_cell_1/MatMul_1/ReadVariableOp(while/gru_cell_1/MatMul_1/ReadVariableOp2B
while/gru_cell_1/ReadVariableOpwhile/gru_cell_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
Æ
 
while_cond_8100
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_12
.while_while_cond_8100___redundant_placeholder02
.while_while_cond_8100___redundant_placeholder12
.while_while_cond_8100___redundant_placeholder22
.while_while_cond_8100___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
 <
À
=__inference_gru_layer_call_and_return_conditional_losses_7603

inputs
gru_cell_7527
gru_cell_7529
gru_cell_7531
identity¢ gru_cell/StatefulPartitionedCall¢whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ü
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_2Ù
 gru_cell/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0gru_cell_7527gru_cell_7529gru_cell_7531*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_gru_cell_layer_call_and_return_conditional_losses_71622"
 gru_cell/StatefulPartitionedCall
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2
TensorArrayV2_1/element_shape¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterÔ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0gru_cell_7527gru_cell_7529gru_cell_7531*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ : : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_7539*
condR
while_cond_7538*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ : : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    22
0TensorArrayV2Stack/TensorListStack/element_shapeñ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm®
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime
IdentityIdentitytranspose_1:y:0!^gru_cell/StatefulPartitionedCall^while*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::2D
 gru_cell/StatefulPartitionedCall gru_cell/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ÏX
Ä
=__inference_gru_layer_call_and_return_conditional_losses_8495

inputs$
 gru_cell_readvariableop_resource+
'gru_cell_matmul_readvariableop_resource-
)gru_cell_matmul_1_readvariableop_resource
identity¢gru_cell/MatMul/ReadVariableOp¢ gru_cell/MatMul_1/ReadVariableOp¢gru_cell/ReadVariableOp¢whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ü
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_2
gru_cell/ReadVariableOpReadVariableOp gru_cell_readvariableop_resource*
_output_shapes

:`*
dtype02
gru_cell/ReadVariableOp
gru_cell/unstackUnpackgru_cell/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num2
gru_cell/unstack¨
gru_cell/MatMul/ReadVariableOpReadVariableOp'gru_cell_matmul_readvariableop_resource*
_output_shapes

:`*
dtype02 
gru_cell/MatMul/ReadVariableOp 
gru_cell/MatMulMatMulstrided_slice_2:output:0&gru_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
gru_cell/MatMul
gru_cell/BiasAddBiasAddgru_cell/MatMul:product:0gru_cell/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
gru_cell/BiasAddb
gru_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru_cell/Const
gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
gru_cell/split/split_dimÐ
gru_cell/splitSplit!gru_cell/split/split_dim:output:0gru_cell/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
gru_cell/split®
 gru_cell/MatMul_1/ReadVariableOpReadVariableOp)gru_cell_matmul_1_readvariableop_resource*
_output_shapes

: `*
dtype02"
 gru_cell/MatMul_1/ReadVariableOp
gru_cell/MatMul_1MatMulzeros:output:0(gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
gru_cell/MatMul_1
gru_cell/BiasAdd_1BiasAddgru_cell/MatMul_1:product:0gru_cell/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
gru_cell/BiasAdd_1y
gru_cell/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"        ÿÿÿÿ2
gru_cell/Const_1
gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
gru_cell/split_1/split_dim
gru_cell/split_1SplitVgru_cell/BiasAdd_1:output:0gru_cell/Const_1:output:0#gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
gru_cell/split_1
gru_cell/addAddV2gru_cell/split:output:0gru_cell/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/adds
gru_cell/SigmoidSigmoidgru_cell/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/Sigmoid
gru_cell/add_1AddV2gru_cell/split:output:1gru_cell/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/add_1y
gru_cell/Sigmoid_1Sigmoidgru_cell/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/Sigmoid_1
gru_cell/mulMulgru_cell/Sigmoid_1:y:0gru_cell/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/mul
gru_cell/add_2AddV2gru_cell/split:output:2gru_cell/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/add_2l
gru_cell/ReluRelugru_cell/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/Relu
gru_cell/mul_1Mulgru_cell/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/mul_1e
gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru_cell/sub/x
gru_cell/subSubgru_cell/sub/x:output:0gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/sub
gru_cell/mul_2Mulgru_cell/sub:z:0gru_cell/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/mul_2
gru_cell/add_3AddV2gru_cell/mul_1:z:0gru_cell/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/add_3
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2
TensorArrayV2_1/element_shape¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0 gru_cell_readvariableop_resource'gru_cell_matmul_readvariableop_resource)gru_cell_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ : : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_8405*
condR
while_cond_8404*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ : : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    22
0TensorArrayV2Stack/TensorListStack/element_shapeè
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¥
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeÍ
IdentityIdentitytranspose_1:y:0^gru_cell/MatMul/ReadVariableOp!^gru_cell/MatMul_1/ReadVariableOp^gru_cell/ReadVariableOp^while*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ:::2@
gru_cell/MatMul/ReadVariableOpgru_cell/MatMul/ReadVariableOp2D
 gru_cell/MatMul_1/ReadVariableOp gru_cell/MatMul_1/ReadVariableOp22
gru_cell/ReadVariableOpgru_cell/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¦Z
Ò
?__inference_gru_1_layer_call_and_return_conditional_losses_8713

inputs&
"gru_cell_1_readvariableop_resource-
)gru_cell_1_matmul_readvariableop_resource/
+gru_cell_1_matmul_1_readvariableop_resource
identity¢ gru_cell_1/MatMul/ReadVariableOp¢"gru_cell_1/MatMul_1/ReadVariableOp¢gru_cell_1/ReadVariableOp¢whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ü
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2
strided_slice_2
gru_cell_1/ReadVariableOpReadVariableOp"gru_cell_1_readvariableop_resource*
_output_shapes

:0*
dtype02
gru_cell_1/ReadVariableOp
gru_cell_1/unstackUnpack!gru_cell_1/ReadVariableOp:value:0*
T0* 
_output_shapes
:0:0*	
num2
gru_cell_1/unstack®
 gru_cell_1/MatMul/ReadVariableOpReadVariableOp)gru_cell_1_matmul_readvariableop_resource*
_output_shapes

: 0*
dtype02"
 gru_cell_1/MatMul/ReadVariableOp¦
gru_cell_1/MatMulMatMulstrided_slice_2:output:0(gru_cell_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02
gru_cell_1/MatMul
gru_cell_1/BiasAddBiasAddgru_cell_1/MatMul:product:0gru_cell_1/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02
gru_cell_1/BiasAddf
gru_cell_1/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru_cell_1/Const
gru_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
gru_cell_1/split/split_dimØ
gru_cell_1/splitSplit#gru_cell_1/split/split_dim:output:0gru_cell_1/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
gru_cell_1/split´
"gru_cell_1/MatMul_1/ReadVariableOpReadVariableOp+gru_cell_1_matmul_1_readvariableop_resource*
_output_shapes

:0*
dtype02$
"gru_cell_1/MatMul_1/ReadVariableOp¢
gru_cell_1/MatMul_1MatMulzeros:output:0*gru_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02
gru_cell_1/MatMul_1¥
gru_cell_1/BiasAdd_1BiasAddgru_cell_1/MatMul_1:product:0gru_cell_1/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02
gru_cell_1/BiasAdd_1}
gru_cell_1/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      ÿÿÿÿ2
gru_cell_1/Const_1
gru_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
gru_cell_1/split_1/split_dim
gru_cell_1/split_1SplitVgru_cell_1/BiasAdd_1:output:0gru_cell_1/Const_1:output:0%gru_cell_1/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
gru_cell_1/split_1
gru_cell_1/addAddV2gru_cell_1/split:output:0gru_cell_1/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_cell_1/addy
gru_cell_1/SigmoidSigmoidgru_cell_1/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_cell_1/Sigmoid
gru_cell_1/add_1AddV2gru_cell_1/split:output:1gru_cell_1/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_cell_1/add_1
gru_cell_1/Sigmoid_1Sigmoidgru_cell_1/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_cell_1/Sigmoid_1
gru_cell_1/mulMulgru_cell_1/Sigmoid_1:y:0gru_cell_1/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_cell_1/mul
gru_cell_1/add_2AddV2gru_cell_1/split:output:2gru_cell_1/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_cell_1/add_2r
gru_cell_1/ReluRelugru_cell_1/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_cell_1/Relu
gru_cell_1/mul_1Mulgru_cell_1/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_cell_1/mul_1i
gru_cell_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru_cell_1/sub/x
gru_cell_1/subSubgru_cell_1/sub/x:output:0gru_cell_1/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_cell_1/sub
gru_cell_1/mul_2Mulgru_cell_1/sub:z:0gru_cell_1/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_cell_1/mul_2
gru_cell_1/add_3AddV2gru_cell_1/mul_1:z:0gru_cell_1/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_cell_1/add_3
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
TensorArrayV2_1/element_shape¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter£
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0"gru_cell_1_readvariableop_resource)gru_cell_1_matmul_readvariableop_resource+gru_cell_1_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_8623*
condR
while_cond_8622*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   22
0TensorArrayV2Stack/TensorListStack/element_shapeè
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¥
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeØ
IdentityIdentitystrided_slice_3:output:0!^gru_cell_1/MatMul/ReadVariableOp#^gru_cell_1/MatMul_1/ReadVariableOp^gru_cell_1/ReadVariableOp^while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ :::2D
 gru_cell_1/MatMul/ReadVariableOp gru_cell_1/MatMul/ReadVariableOp2H
"gru_cell_1/MatMul_1/ReadVariableOp"gru_cell_1/MatMul_1/ReadVariableOp26
gru_cell_1/ReadVariableOpgru_cell_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ÃP
×
gru_1_while_body_9669(
$gru_1_while_gru_1_while_loop_counter.
*gru_1_while_gru_1_while_maximum_iterations
gru_1_while_placeholder
gru_1_while_placeholder_1
gru_1_while_placeholder_2'
#gru_1_while_gru_1_strided_slice_1_0c
_gru_1_while_tensorarrayv2read_tensorlistgetitem_gru_1_tensorarrayunstack_tensorlistfromtensor_04
0gru_1_while_gru_cell_1_readvariableop_resource_0;
7gru_1_while_gru_cell_1_matmul_readvariableop_resource_0=
9gru_1_while_gru_cell_1_matmul_1_readvariableop_resource_0
gru_1_while_identity
gru_1_while_identity_1
gru_1_while_identity_2
gru_1_while_identity_3
gru_1_while_identity_4%
!gru_1_while_gru_1_strided_slice_1a
]gru_1_while_tensorarrayv2read_tensorlistgetitem_gru_1_tensorarrayunstack_tensorlistfromtensor2
.gru_1_while_gru_cell_1_readvariableop_resource9
5gru_1_while_gru_cell_1_matmul_readvariableop_resource;
7gru_1_while_gru_cell_1_matmul_1_readvariableop_resource¢,gru_1/while/gru_cell_1/MatMul/ReadVariableOp¢.gru_1/while/gru_cell_1/MatMul_1/ReadVariableOp¢%gru_1/while/gru_cell_1/ReadVariableOpÏ
=gru_1/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2?
=gru_1/while/TensorArrayV2Read/TensorListGetItem/element_shape÷
/gru_1/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem_gru_1_while_tensorarrayv2read_tensorlistgetitem_gru_1_tensorarrayunstack_tensorlistfromtensor_0gru_1_while_placeholderFgru_1/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype021
/gru_1/while/TensorArrayV2Read/TensorListGetItem¿
%gru_1/while/gru_cell_1/ReadVariableOpReadVariableOp0gru_1_while_gru_cell_1_readvariableop_resource_0*
_output_shapes

:0*
dtype02'
%gru_1/while/gru_cell_1/ReadVariableOp¯
gru_1/while/gru_cell_1/unstackUnpack-gru_1/while/gru_cell_1/ReadVariableOp:value:0*
T0* 
_output_shapes
:0:0*	
num2 
gru_1/while/gru_cell_1/unstackÔ
,gru_1/while/gru_cell_1/MatMul/ReadVariableOpReadVariableOp7gru_1_while_gru_cell_1_matmul_readvariableop_resource_0*
_output_shapes

: 0*
dtype02.
,gru_1/while/gru_cell_1/MatMul/ReadVariableOpè
gru_1/while/gru_cell_1/MatMulMatMul6gru_1/while/TensorArrayV2Read/TensorListGetItem:item:04gru_1/while/gru_cell_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02
gru_1/while/gru_cell_1/MatMulÏ
gru_1/while/gru_cell_1/BiasAddBiasAdd'gru_1/while/gru_cell_1/MatMul:product:0'gru_1/while/gru_cell_1/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02 
gru_1/while/gru_cell_1/BiasAdd~
gru_1/while/gru_cell_1/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru_1/while/gru_cell_1/Const
&gru_1/while/gru_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2(
&gru_1/while/gru_cell_1/split/split_dim
gru_1/while/gru_cell_1/splitSplit/gru_1/while/gru_cell_1/split/split_dim:output:0'gru_1/while/gru_cell_1/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
gru_1/while/gru_cell_1/splitÚ
.gru_1/while/gru_cell_1/MatMul_1/ReadVariableOpReadVariableOp9gru_1_while_gru_cell_1_matmul_1_readvariableop_resource_0*
_output_shapes

:0*
dtype020
.gru_1/while/gru_cell_1/MatMul_1/ReadVariableOpÑ
gru_1/while/gru_cell_1/MatMul_1MatMulgru_1_while_placeholder_26gru_1/while/gru_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02!
gru_1/while/gru_cell_1/MatMul_1Õ
 gru_1/while/gru_cell_1/BiasAdd_1BiasAdd)gru_1/while/gru_cell_1/MatMul_1:product:0'gru_1/while/gru_cell_1/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02"
 gru_1/while/gru_cell_1/BiasAdd_1
gru_1/while/gru_cell_1/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      ÿÿÿÿ2 
gru_1/while/gru_cell_1/Const_1
(gru_1/while/gru_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2*
(gru_1/while/gru_cell_1/split_1/split_dimÆ
gru_1/while/gru_cell_1/split_1SplitV)gru_1/while/gru_cell_1/BiasAdd_1:output:0'gru_1/while/gru_cell_1/Const_1:output:01gru_1/while/gru_cell_1/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2 
gru_1/while/gru_cell_1/split_1Ã
gru_1/while/gru_cell_1/addAddV2%gru_1/while/gru_cell_1/split:output:0'gru_1/while/gru_cell_1/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_1/while/gru_cell_1/add
gru_1/while/gru_cell_1/SigmoidSigmoidgru_1/while/gru_cell_1/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
gru_1/while/gru_cell_1/SigmoidÇ
gru_1/while/gru_cell_1/add_1AddV2%gru_1/while/gru_cell_1/split:output:1'gru_1/while/gru_cell_1/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_1/while/gru_cell_1/add_1£
 gru_1/while/gru_cell_1/Sigmoid_1Sigmoid gru_1/while/gru_cell_1/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 gru_1/while/gru_cell_1/Sigmoid_1À
gru_1/while/gru_cell_1/mulMul$gru_1/while/gru_cell_1/Sigmoid_1:y:0'gru_1/while/gru_cell_1/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_1/while/gru_cell_1/mul¾
gru_1/while/gru_cell_1/add_2AddV2%gru_1/while/gru_cell_1/split:output:2gru_1/while/gru_cell_1/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_1/while/gru_cell_1/add_2
gru_1/while/gru_cell_1/ReluRelu gru_1/while/gru_cell_1/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_1/while/gru_cell_1/Relu´
gru_1/while/gru_cell_1/mul_1Mul"gru_1/while/gru_cell_1/Sigmoid:y:0gru_1_while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_1/while/gru_cell_1/mul_1
gru_1/while/gru_cell_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru_1/while/gru_cell_1/sub/x¼
gru_1/while/gru_cell_1/subSub%gru_1/while/gru_cell_1/sub/x:output:0"gru_1/while/gru_cell_1/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_1/while/gru_cell_1/subÀ
gru_1/while/gru_cell_1/mul_2Mulgru_1/while/gru_cell_1/sub:z:0)gru_1/while/gru_cell_1/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_1/while/gru_cell_1/mul_2»
gru_1/while/gru_cell_1/add_3AddV2 gru_1/while/gru_cell_1/mul_1:z:0 gru_1/while/gru_cell_1/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_1/while/gru_cell_1/add_3ü
0gru_1/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemgru_1_while_placeholder_1gru_1_while_placeholder gru_1/while/gru_cell_1/add_3:z:0*
_output_shapes
: *
element_dtype022
0gru_1/while/TensorArrayV2Write/TensorListSetItemh
gru_1/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
gru_1/while/add/y
gru_1/while/addAddV2gru_1_while_placeholdergru_1/while/add/y:output:0*
T0*
_output_shapes
: 2
gru_1/while/addl
gru_1/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
gru_1/while/add_1/y
gru_1/while/add_1AddV2$gru_1_while_gru_1_while_loop_countergru_1/while/add_1/y:output:0*
T0*
_output_shapes
: 2
gru_1/while/add_1ø
gru_1/while/IdentityIdentitygru_1/while/add_1:z:0-^gru_1/while/gru_cell_1/MatMul/ReadVariableOp/^gru_1/while/gru_cell_1/MatMul_1/ReadVariableOp&^gru_1/while/gru_cell_1/ReadVariableOp*
T0*
_output_shapes
: 2
gru_1/while/Identity
gru_1/while/Identity_1Identity*gru_1_while_gru_1_while_maximum_iterations-^gru_1/while/gru_cell_1/MatMul/ReadVariableOp/^gru_1/while/gru_cell_1/MatMul_1/ReadVariableOp&^gru_1/while/gru_cell_1/ReadVariableOp*
T0*
_output_shapes
: 2
gru_1/while/Identity_1ú
gru_1/while/Identity_2Identitygru_1/while/add:z:0-^gru_1/while/gru_cell_1/MatMul/ReadVariableOp/^gru_1/while/gru_cell_1/MatMul_1/ReadVariableOp&^gru_1/while/gru_cell_1/ReadVariableOp*
T0*
_output_shapes
: 2
gru_1/while/Identity_2§
gru_1/while/Identity_3Identity@gru_1/while/TensorArrayV2Write/TensorListSetItem:output_handle:0-^gru_1/while/gru_cell_1/MatMul/ReadVariableOp/^gru_1/while/gru_cell_1/MatMul_1/ReadVariableOp&^gru_1/while/gru_cell_1/ReadVariableOp*
T0*
_output_shapes
: 2
gru_1/while/Identity_3
gru_1/while/Identity_4Identity gru_1/while/gru_cell_1/add_3:z:0-^gru_1/while/gru_cell_1/MatMul/ReadVariableOp/^gru_1/while/gru_cell_1/MatMul_1/ReadVariableOp&^gru_1/while/gru_cell_1/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_1/while/Identity_4"H
!gru_1_while_gru_1_strided_slice_1#gru_1_while_gru_1_strided_slice_1_0"t
7gru_1_while_gru_cell_1_matmul_1_readvariableop_resource9gru_1_while_gru_cell_1_matmul_1_readvariableop_resource_0"p
5gru_1_while_gru_cell_1_matmul_readvariableop_resource7gru_1_while_gru_cell_1_matmul_readvariableop_resource_0"b
.gru_1_while_gru_cell_1_readvariableop_resource0gru_1_while_gru_cell_1_readvariableop_resource_0"5
gru_1_while_identitygru_1/while/Identity:output:0"9
gru_1_while_identity_1gru_1/while/Identity_1:output:0"9
gru_1_while_identity_2gru_1/while/Identity_2:output:0"9
gru_1_while_identity_3gru_1/while/Identity_3:output:0"9
gru_1_while_identity_4gru_1/while/Identity_4:output:0"À
]gru_1_while_tensorarrayv2read_tensorlistgetitem_gru_1_tensorarrayunstack_tensorlistfromtensor_gru_1_while_tensorarrayv2read_tensorlistgetitem_gru_1_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :ÿÿÿÿÿÿÿÿÿ: : :::2\
,gru_1/while/gru_cell_1/MatMul/ReadVariableOp,gru_1/while/gru_cell_1/MatMul/ReadVariableOp2`
.gru_1/while/gru_cell_1/MatMul_1/ReadVariableOp.gru_1/while/gru_cell_1/MatMul_1/ReadVariableOp2N
%gru_1/while/gru_cell_1/ReadVariableOp%gru_1/while/gru_cell_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
Ë
¥
while_cond_11082
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_11082___redundant_placeholder03
/while_while_cond_11082___redundant_placeholder13
/while_while_cond_11082___redundant_placeholder23
/while_while_cond_11082___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
üD

while_body_10036
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0,
(while_gru_cell_readvariableop_resource_03
/while_gru_cell_matmul_readvariableop_resource_05
1while_gru_cell_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor*
&while_gru_cell_readvariableop_resource1
-while_gru_cell_matmul_readvariableop_resource3
/while_gru_cell_matmul_1_readvariableop_resource¢$while/gru_cell/MatMul/ReadVariableOp¢&while/gru_cell/MatMul_1/ReadVariableOp¢while/gru_cell/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÓ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem§
while/gru_cell/ReadVariableOpReadVariableOp(while_gru_cell_readvariableop_resource_0*
_output_shapes

:`*
dtype02
while/gru_cell/ReadVariableOp
while/gru_cell/unstackUnpack%while/gru_cell/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num2
while/gru_cell/unstack¼
$while/gru_cell/MatMul/ReadVariableOpReadVariableOp/while_gru_cell_matmul_readvariableop_resource_0*
_output_shapes

:`*
dtype02&
$while/gru_cell/MatMul/ReadVariableOpÊ
while/gru_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0,while/gru_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
while/gru_cell/MatMul¯
while/gru_cell/BiasAddBiasAddwhile/gru_cell/MatMul:product:0while/gru_cell/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
while/gru_cell/BiasAddn
while/gru_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/gru_cell/Const
while/gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2 
while/gru_cell/split/split_dimè
while/gru_cell/splitSplit'while/gru_cell/split/split_dim:output:0while/gru_cell/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
while/gru_cell/splitÂ
&while/gru_cell/MatMul_1/ReadVariableOpReadVariableOp1while_gru_cell_matmul_1_readvariableop_resource_0*
_output_shapes

: `*
dtype02(
&while/gru_cell/MatMul_1/ReadVariableOp³
while/gru_cell/MatMul_1MatMulwhile_placeholder_2.while/gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
while/gru_cell/MatMul_1µ
while/gru_cell/BiasAdd_1BiasAdd!while/gru_cell/MatMul_1:product:0while/gru_cell/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
while/gru_cell/BiasAdd_1
while/gru_cell/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"        ÿÿÿÿ2
while/gru_cell/Const_1
 while/gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2"
 while/gru_cell/split_1/split_dim
while/gru_cell/split_1SplitV!while/gru_cell/BiasAdd_1:output:0while/gru_cell/Const_1:output:0)while/gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
while/gru_cell/split_1£
while/gru_cell/addAddV2while/gru_cell/split:output:0while/gru_cell/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/add
while/gru_cell/SigmoidSigmoidwhile/gru_cell/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/Sigmoid§
while/gru_cell/add_1AddV2while/gru_cell/split:output:1while/gru_cell/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/add_1
while/gru_cell/Sigmoid_1Sigmoidwhile/gru_cell/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/Sigmoid_1 
while/gru_cell/mulMulwhile/gru_cell/Sigmoid_1:y:0while/gru_cell/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/mul
while/gru_cell/add_2AddV2while/gru_cell/split:output:2while/gru_cell/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/add_2~
while/gru_cell/ReluReluwhile/gru_cell/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/Relu
while/gru_cell/mul_1Mulwhile/gru_cell/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/mul_1q
while/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
while/gru_cell/sub/x
while/gru_cell/subSubwhile/gru_cell/sub/x:output:0while/gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/sub 
while/gru_cell/mul_2Mulwhile/gru_cell/sub:z:0!while/gru_cell/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/mul_2
while/gru_cell/add_3AddV2while/gru_cell/mul_1:z:0while/gru_cell/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/add_3Ü
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell/add_3:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1Î
while/IdentityIdentitywhile/add_1:z:0%^while/gru_cell/MatMul/ReadVariableOp'^while/gru_cell/MatMul_1/ReadVariableOp^while/gru_cell/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identityá
while/Identity_1Identitywhile_while_maximum_iterations%^while/gru_cell/MatMul/ReadVariableOp'^while/gru_cell/MatMul_1/ReadVariableOp^while/gru_cell/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1Ð
while/Identity_2Identitywhile/add:z:0%^while/gru_cell/MatMul/ReadVariableOp'^while/gru_cell/MatMul_1/ReadVariableOp^while/gru_cell/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2ý
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0%^while/gru_cell/MatMul/ReadVariableOp'^while/gru_cell/MatMul_1/ReadVariableOp^while/gru_cell/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3ì
while/Identity_4Identitywhile/gru_cell/add_3:z:0%^while/gru_cell/MatMul/ReadVariableOp'^while/gru_cell/MatMul_1/ReadVariableOp^while/gru_cell/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4"d
/while_gru_cell_matmul_1_readvariableop_resource1while_gru_cell_matmul_1_readvariableop_resource_0"`
-while_gru_cell_matmul_readvariableop_resource/while_gru_cell_matmul_readvariableop_resource_0"R
&while_gru_cell_readvariableop_resource(while_gru_cell_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :ÿÿÿÿÿÿÿÿÿ : : :::2L
$while/gru_cell/MatMul/ReadVariableOp$while/gru_cell/MatMul/ReadVariableOp2P
&while/gru_cell/MatMul_1/ReadVariableOp&while/gru_cell/MatMul_1/ReadVariableOp2>
while/gru_cell/ReadVariableOpwhile/gru_cell/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
: 
Æ
 
while_cond_8404
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_12
.while_while_cond_8404___redundant_placeholder02
.while_while_cond_8404___redundant_placeholder12
.while_while_cond_8404___redundant_placeholder22
.while_while_cond_8404___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
:
õ

gru_1_while_cond_9339(
$gru_1_while_gru_1_while_loop_counter.
*gru_1_while_gru_1_while_maximum_iterations
gru_1_while_placeholder
gru_1_while_placeholder_1
gru_1_while_placeholder_2*
&gru_1_while_less_gru_1_strided_slice_1>
:gru_1_while_gru_1_while_cond_9339___redundant_placeholder0>
:gru_1_while_gru_1_while_cond_9339___redundant_placeholder1>
:gru_1_while_gru_1_while_cond_9339___redundant_placeholder2>
:gru_1_while_gru_1_while_cond_9339___redundant_placeholder3
gru_1_while_identity

gru_1/while/LessLessgru_1_while_placeholder&gru_1_while_less_gru_1_strided_slice_1*
T0*
_output_shapes
: 2
gru_1/while/Lesso
gru_1/while/IdentityIdentitygru_1/while/Less:z:0*
T0
*
_output_shapes
: 2
gru_1/while/Identity"5
gru_1_while_identitygru_1/while/Identity:output:0*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
Æ
 
while_cond_8781
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_12
.while_while_cond_8781___redundant_placeholder02
.while_while_cond_8781___redundant_placeholder12
.while_while_cond_8781___redundant_placeholder22
.while_while_cond_8781___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
Ç
b
D__inference_dropout_1_layer_call_and_return_conditional_losses_11212

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ã

D__inference_sequential_layer_call_and_return_conditional_losses_8959
	gru_input
gru_8518
gru_8520
gru_8522

gru_1_8895

gru_1_8897

gru_1_8899

dense_8953

dense_8955
identity¢dense/StatefulPartitionedCall¢dropout/StatefulPartitionedCall¢!dropout_1/StatefulPartitionedCall¢gru/StatefulPartitionedCall¢gru_1/StatefulPartitionedCall
gru/StatefulPartitionedCallStatefulPartitionedCall	gru_inputgru_8518gru_8520gru_8522*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *F
fAR?
=__inference_gru_layer_call_and_return_conditional_losses_83362
gru/StatefulPartitionedCall
dropout/StatefulPartitionedCallStatefulPartitionedCall$gru/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_85372!
dropout/StatefulPartitionedCall²
gru_1/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0
gru_1_8895
gru_1_8897
gru_1_8899*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_gru_1_layer_call_and_return_conditional_losses_87132
gru_1/StatefulPartitionedCall²
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall&gru_1/StatefulPartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dropout_1_layer_call_and_return_conditional_losses_89142#
!dropout_1/StatefulPartitionedCall¦
dense/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0
dense_8953
dense_8955*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_89422
dense/StatefulPartitionedCall
IdentityIdentity&dense/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall^gru/StatefulPartitionedCall^gru_1/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:ÿÿÿÿÿÿÿÿÿ::::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2:
gru/StatefulPartitionedCallgru/StatefulPartitionedCall2>
gru_1/StatefulPartitionedCallgru_1/StatefulPartitionedCall:V R
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#
_user_specified_name	gru_input
üD

while_body_10217
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0,
(while_gru_cell_readvariableop_resource_03
/while_gru_cell_matmul_readvariableop_resource_05
1while_gru_cell_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor*
&while_gru_cell_readvariableop_resource1
-while_gru_cell_matmul_readvariableop_resource3
/while_gru_cell_matmul_1_readvariableop_resource¢$while/gru_cell/MatMul/ReadVariableOp¢&while/gru_cell/MatMul_1/ReadVariableOp¢while/gru_cell/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÓ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem§
while/gru_cell/ReadVariableOpReadVariableOp(while_gru_cell_readvariableop_resource_0*
_output_shapes

:`*
dtype02
while/gru_cell/ReadVariableOp
while/gru_cell/unstackUnpack%while/gru_cell/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num2
while/gru_cell/unstack¼
$while/gru_cell/MatMul/ReadVariableOpReadVariableOp/while_gru_cell_matmul_readvariableop_resource_0*
_output_shapes

:`*
dtype02&
$while/gru_cell/MatMul/ReadVariableOpÊ
while/gru_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0,while/gru_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
while/gru_cell/MatMul¯
while/gru_cell/BiasAddBiasAddwhile/gru_cell/MatMul:product:0while/gru_cell/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
while/gru_cell/BiasAddn
while/gru_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/gru_cell/Const
while/gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2 
while/gru_cell/split/split_dimè
while/gru_cell/splitSplit'while/gru_cell/split/split_dim:output:0while/gru_cell/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
while/gru_cell/splitÂ
&while/gru_cell/MatMul_1/ReadVariableOpReadVariableOp1while_gru_cell_matmul_1_readvariableop_resource_0*
_output_shapes

: `*
dtype02(
&while/gru_cell/MatMul_1/ReadVariableOp³
while/gru_cell/MatMul_1MatMulwhile_placeholder_2.while/gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
while/gru_cell/MatMul_1µ
while/gru_cell/BiasAdd_1BiasAdd!while/gru_cell/MatMul_1:product:0while/gru_cell/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
while/gru_cell/BiasAdd_1
while/gru_cell/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"        ÿÿÿÿ2
while/gru_cell/Const_1
 while/gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2"
 while/gru_cell/split_1/split_dim
while/gru_cell/split_1SplitV!while/gru_cell/BiasAdd_1:output:0while/gru_cell/Const_1:output:0)while/gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
while/gru_cell/split_1£
while/gru_cell/addAddV2while/gru_cell/split:output:0while/gru_cell/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/add
while/gru_cell/SigmoidSigmoidwhile/gru_cell/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/Sigmoid§
while/gru_cell/add_1AddV2while/gru_cell/split:output:1while/gru_cell/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/add_1
while/gru_cell/Sigmoid_1Sigmoidwhile/gru_cell/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/Sigmoid_1 
while/gru_cell/mulMulwhile/gru_cell/Sigmoid_1:y:0while/gru_cell/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/mul
while/gru_cell/add_2AddV2while/gru_cell/split:output:2while/gru_cell/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/add_2~
while/gru_cell/ReluReluwhile/gru_cell/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/Relu
while/gru_cell/mul_1Mulwhile/gru_cell/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/mul_1q
while/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
while/gru_cell/sub/x
while/gru_cell/subSubwhile/gru_cell/sub/x:output:0while/gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/sub 
while/gru_cell/mul_2Mulwhile/gru_cell/sub:z:0!while/gru_cell/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/mul_2
while/gru_cell/add_3AddV2while/gru_cell/mul_1:z:0while/gru_cell/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/add_3Ü
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell/add_3:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1Î
while/IdentityIdentitywhile/add_1:z:0%^while/gru_cell/MatMul/ReadVariableOp'^while/gru_cell/MatMul_1/ReadVariableOp^while/gru_cell/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identityá
while/Identity_1Identitywhile_while_maximum_iterations%^while/gru_cell/MatMul/ReadVariableOp'^while/gru_cell/MatMul_1/ReadVariableOp^while/gru_cell/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1Ð
while/Identity_2Identitywhile/add:z:0%^while/gru_cell/MatMul/ReadVariableOp'^while/gru_cell/MatMul_1/ReadVariableOp^while/gru_cell/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2ý
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0%^while/gru_cell/MatMul/ReadVariableOp'^while/gru_cell/MatMul_1/ReadVariableOp^while/gru_cell/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3ì
while/Identity_4Identitywhile/gru_cell/add_3:z:0%^while/gru_cell/MatMul/ReadVariableOp'^while/gru_cell/MatMul_1/ReadVariableOp^while/gru_cell/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4"d
/while_gru_cell_matmul_1_readvariableop_resource1while_gru_cell_matmul_1_readvariableop_resource_0"`
-while_gru_cell_matmul_readvariableop_resource/while_gru_cell_matmul_readvariableop_resource_0"R
&while_gru_cell_readvariableop_resource(while_gru_cell_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :ÿÿÿÿÿÿÿÿÿ : : :::2L
$while/gru_cell/MatMul/ReadVariableOp$while/gru_cell/MatMul/ReadVariableOp2P
&while/gru_cell/MatMul_1/ReadVariableOp&while/gru_cell/MatMul_1/ReadVariableOp2>
while/gru_cell/ReadVariableOpwhile/gru_cell/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
: 
Í

D__inference_sequential_layer_call_and_return_conditional_losses_9766

inputs(
$gru_gru_cell_readvariableop_resource/
+gru_gru_cell_matmul_readvariableop_resource1
-gru_gru_cell_matmul_1_readvariableop_resource,
(gru_1_gru_cell_1_readvariableop_resource3
/gru_1_gru_cell_1_matmul_readvariableop_resource5
1gru_1_gru_cell_1_matmul_1_readvariableop_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource
identity¢dense/BiasAdd/ReadVariableOp¢dense/MatMul/ReadVariableOp¢"gru/gru_cell/MatMul/ReadVariableOp¢$gru/gru_cell/MatMul_1/ReadVariableOp¢gru/gru_cell/ReadVariableOp¢	gru/while¢&gru_1/gru_cell_1/MatMul/ReadVariableOp¢(gru_1/gru_cell_1/MatMul_1/ReadVariableOp¢gru_1/gru_cell_1/ReadVariableOp¢gru_1/whileL
	gru/ShapeShapeinputs*
T0*
_output_shapes
:2
	gru/Shape|
gru/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru/strided_slice/stack
gru/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
gru/strided_slice/stack_1
gru/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru/strided_slice/stack_2ú
gru/strided_sliceStridedSlicegru/Shape:output:0 gru/strided_slice/stack:output:0"gru/strided_slice/stack_1:output:0"gru/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
gru/strided_sliced
gru/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
gru/zeros/mul/y|
gru/zeros/mulMulgru/strided_slice:output:0gru/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
gru/zeros/mulg
gru/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
gru/zeros/Less/yw
gru/zeros/LessLessgru/zeros/mul:z:0gru/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
gru/zeros/Lessj
gru/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
gru/zeros/packed/1
gru/zeros/packedPackgru/strided_slice:output:0gru/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
gru/zeros/packedg
gru/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
gru/zeros/Const
	gru/zerosFillgru/zeros/packed:output:0gru/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
	gru/zeros}
gru/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
gru/transpose/perm
gru/transpose	Transposeinputsgru/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru/transpose[
gru/Shape_1Shapegru/transpose:y:0*
T0*
_output_shapes
:2
gru/Shape_1
gru/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru/strided_slice_1/stack
gru/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
gru/strided_slice_1/stack_1
gru/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru/strided_slice_1/stack_2
gru/strided_slice_1StridedSlicegru/Shape_1:output:0"gru/strided_slice_1/stack:output:0$gru/strided_slice_1/stack_1:output:0$gru/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
gru/strided_slice_1
gru/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2!
gru/TensorArrayV2/element_shapeÂ
gru/TensorArrayV2TensorListReserve(gru/TensorArrayV2/element_shape:output:0gru/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
gru/TensorArrayV2Ç
9gru/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2;
9gru/TensorArrayUnstack/TensorListFromTensor/element_shape
+gru/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorgru/transpose:y:0Bgru/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02-
+gru/TensorArrayUnstack/TensorListFromTensor
gru/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru/strided_slice_2/stack
gru/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
gru/strided_slice_2/stack_1
gru/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru/strided_slice_2/stack_2
gru/strided_slice_2StridedSlicegru/transpose:y:0"gru/strided_slice_2/stack:output:0$gru/strided_slice_2/stack_1:output:0$gru/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
gru/strided_slice_2
gru/gru_cell/ReadVariableOpReadVariableOp$gru_gru_cell_readvariableop_resource*
_output_shapes

:`*
dtype02
gru/gru_cell/ReadVariableOp
gru/gru_cell/unstackUnpack#gru/gru_cell/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num2
gru/gru_cell/unstack´
"gru/gru_cell/MatMul/ReadVariableOpReadVariableOp+gru_gru_cell_matmul_readvariableop_resource*
_output_shapes

:`*
dtype02$
"gru/gru_cell/MatMul/ReadVariableOp°
gru/gru_cell/MatMulMatMulgru/strided_slice_2:output:0*gru/gru_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
gru/gru_cell/MatMul§
gru/gru_cell/BiasAddBiasAddgru/gru_cell/MatMul:product:0gru/gru_cell/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
gru/gru_cell/BiasAddj
gru/gru_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru/gru_cell/Const
gru/gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
gru/gru_cell/split/split_dimà
gru/gru_cell/splitSplit%gru/gru_cell/split/split_dim:output:0gru/gru_cell/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
gru/gru_cell/splitº
$gru/gru_cell/MatMul_1/ReadVariableOpReadVariableOp-gru_gru_cell_matmul_1_readvariableop_resource*
_output_shapes

: `*
dtype02&
$gru/gru_cell/MatMul_1/ReadVariableOp¬
gru/gru_cell/MatMul_1MatMulgru/zeros:output:0,gru/gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
gru/gru_cell/MatMul_1­
gru/gru_cell/BiasAdd_1BiasAddgru/gru_cell/MatMul_1:product:0gru/gru_cell/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
gru/gru_cell/BiasAdd_1
gru/gru_cell/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"        ÿÿÿÿ2
gru/gru_cell/Const_1
gru/gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2 
gru/gru_cell/split_1/split_dim
gru/gru_cell/split_1SplitVgru/gru_cell/BiasAdd_1:output:0gru/gru_cell/Const_1:output:0'gru/gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
gru/gru_cell/split_1
gru/gru_cell/addAddV2gru/gru_cell/split:output:0gru/gru_cell/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/gru_cell/add
gru/gru_cell/SigmoidSigmoidgru/gru_cell/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/gru_cell/Sigmoid
gru/gru_cell/add_1AddV2gru/gru_cell/split:output:1gru/gru_cell/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/gru_cell/add_1
gru/gru_cell/Sigmoid_1Sigmoidgru/gru_cell/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/gru_cell/Sigmoid_1
gru/gru_cell/mulMulgru/gru_cell/Sigmoid_1:y:0gru/gru_cell/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/gru_cell/mul
gru/gru_cell/add_2AddV2gru/gru_cell/split:output:2gru/gru_cell/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/gru_cell/add_2x
gru/gru_cell/ReluRelugru/gru_cell/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/gru_cell/Relu
gru/gru_cell/mul_1Mulgru/gru_cell/Sigmoid:y:0gru/zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/gru_cell/mul_1m
gru/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru/gru_cell/sub/x
gru/gru_cell/subSubgru/gru_cell/sub/x:output:0gru/gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/gru_cell/sub
gru/gru_cell/mul_2Mulgru/gru_cell/sub:z:0gru/gru_cell/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/gru_cell/mul_2
gru/gru_cell/add_3AddV2gru/gru_cell/mul_1:z:0gru/gru_cell/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/gru_cell/add_3
!gru/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2#
!gru/TensorArrayV2_1/element_shapeÈ
gru/TensorArrayV2_1TensorListReserve*gru/TensorArrayV2_1/element_shape:output:0gru/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
gru/TensorArrayV2_1V
gru/timeConst*
_output_shapes
: *
dtype0*
value	B : 2

gru/time
gru/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
gru/while/maximum_iterationsr
gru/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
gru/while/loop_counterÕ
	gru/whileWhilegru/while/loop_counter:output:0%gru/while/maximum_iterations:output:0gru/time:output:0gru/TensorArrayV2_1:handle:0gru/zeros:output:0gru/strided_slice_1:output:0;gru/TensorArrayUnstack/TensorListFromTensor:output_handle:0$gru_gru_cell_readvariableop_resource+gru_gru_cell_matmul_readvariableop_resource-gru_gru_cell_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ : : : : : *%
_read_only_resource_inputs
	*
bodyR
gru_while_body_9513*
condR
gru_while_cond_9512*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ : : : : : *
parallel_iterations 2
	gru/while½
4gru/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    26
4gru/TensorArrayV2Stack/TensorListStack/element_shapeø
&gru/TensorArrayV2Stack/TensorListStackTensorListStackgru/while:output:3=gru/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype02(
&gru/TensorArrayV2Stack/TensorListStack
gru/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
gru/strided_slice_3/stack
gru/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
gru/strided_slice_3/stack_1
gru/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru/strided_slice_3/stack_2²
gru/strided_slice_3StridedSlice/gru/TensorArrayV2Stack/TensorListStack:tensor:0"gru/strided_slice_3/stack:output:0$gru/strided_slice_3/stack_1:output:0$gru/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2
gru/strided_slice_3
gru/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
gru/transpose_1/permµ
gru/transpose_1	Transpose/gru/TensorArrayV2Stack/TensorListStack:tensor:0gru/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/transpose_1n
gru/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
gru/runtime{
dropout/IdentityIdentitygru/transpose_1:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dropout/Identityc
gru_1/ShapeShapedropout/Identity:output:0*
T0*
_output_shapes
:2
gru_1/Shape
gru_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru_1/strided_slice/stack
gru_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
gru_1/strided_slice/stack_1
gru_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru_1/strided_slice/stack_2
gru_1/strided_sliceStridedSlicegru_1/Shape:output:0"gru_1/strided_slice/stack:output:0$gru_1/strided_slice/stack_1:output:0$gru_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
gru_1/strided_sliceh
gru_1/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
gru_1/zeros/mul/y
gru_1/zeros/mulMulgru_1/strided_slice:output:0gru_1/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
gru_1/zeros/mulk
gru_1/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
gru_1/zeros/Less/y
gru_1/zeros/LessLessgru_1/zeros/mul:z:0gru_1/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
gru_1/zeros/Lessn
gru_1/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
gru_1/zeros/packed/1
gru_1/zeros/packedPackgru_1/strided_slice:output:0gru_1/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
gru_1/zeros/packedk
gru_1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
gru_1/zeros/Const
gru_1/zerosFillgru_1/zeros/packed:output:0gru_1/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_1/zeros
gru_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
gru_1/transpose/perm
gru_1/transpose	Transposedropout/Identity:output:0gru_1/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_1/transposea
gru_1/Shape_1Shapegru_1/transpose:y:0*
T0*
_output_shapes
:2
gru_1/Shape_1
gru_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru_1/strided_slice_1/stack
gru_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
gru_1/strided_slice_1/stack_1
gru_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru_1/strided_slice_1/stack_2
gru_1/strided_slice_1StridedSlicegru_1/Shape_1:output:0$gru_1/strided_slice_1/stack:output:0&gru_1/strided_slice_1/stack_1:output:0&gru_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
gru_1/strided_slice_1
!gru_1/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2#
!gru_1/TensorArrayV2/element_shapeÊ
gru_1/TensorArrayV2TensorListReserve*gru_1/TensorArrayV2/element_shape:output:0gru_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
gru_1/TensorArrayV2Ë
;gru_1/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2=
;gru_1/TensorArrayUnstack/TensorListFromTensor/element_shape
-gru_1/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorgru_1/transpose:y:0Dgru_1/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02/
-gru_1/TensorArrayUnstack/TensorListFromTensor
gru_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru_1/strided_slice_2/stack
gru_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
gru_1/strided_slice_2/stack_1
gru_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru_1/strided_slice_2/stack_2 
gru_1/strided_slice_2StridedSlicegru_1/transpose:y:0$gru_1/strided_slice_2/stack:output:0&gru_1/strided_slice_2/stack_1:output:0&gru_1/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2
gru_1/strided_slice_2«
gru_1/gru_cell_1/ReadVariableOpReadVariableOp(gru_1_gru_cell_1_readvariableop_resource*
_output_shapes

:0*
dtype02!
gru_1/gru_cell_1/ReadVariableOp
gru_1/gru_cell_1/unstackUnpack'gru_1/gru_cell_1/ReadVariableOp:value:0*
T0* 
_output_shapes
:0:0*	
num2
gru_1/gru_cell_1/unstackÀ
&gru_1/gru_cell_1/MatMul/ReadVariableOpReadVariableOp/gru_1_gru_cell_1_matmul_readvariableop_resource*
_output_shapes

: 0*
dtype02(
&gru_1/gru_cell_1/MatMul/ReadVariableOp¾
gru_1/gru_cell_1/MatMulMatMulgru_1/strided_slice_2:output:0.gru_1/gru_cell_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02
gru_1/gru_cell_1/MatMul·
gru_1/gru_cell_1/BiasAddBiasAdd!gru_1/gru_cell_1/MatMul:product:0!gru_1/gru_cell_1/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02
gru_1/gru_cell_1/BiasAddr
gru_1/gru_cell_1/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru_1/gru_cell_1/Const
 gru_1/gru_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2"
 gru_1/gru_cell_1/split/split_dimð
gru_1/gru_cell_1/splitSplit)gru_1/gru_cell_1/split/split_dim:output:0!gru_1/gru_cell_1/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
gru_1/gru_cell_1/splitÆ
(gru_1/gru_cell_1/MatMul_1/ReadVariableOpReadVariableOp1gru_1_gru_cell_1_matmul_1_readvariableop_resource*
_output_shapes

:0*
dtype02*
(gru_1/gru_cell_1/MatMul_1/ReadVariableOpº
gru_1/gru_cell_1/MatMul_1MatMulgru_1/zeros:output:00gru_1/gru_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02
gru_1/gru_cell_1/MatMul_1½
gru_1/gru_cell_1/BiasAdd_1BiasAdd#gru_1/gru_cell_1/MatMul_1:product:0!gru_1/gru_cell_1/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02
gru_1/gru_cell_1/BiasAdd_1
gru_1/gru_cell_1/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      ÿÿÿÿ2
gru_1/gru_cell_1/Const_1
"gru_1/gru_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2$
"gru_1/gru_cell_1/split_1/split_dim¨
gru_1/gru_cell_1/split_1SplitV#gru_1/gru_cell_1/BiasAdd_1:output:0!gru_1/gru_cell_1/Const_1:output:0+gru_1/gru_cell_1/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
gru_1/gru_cell_1/split_1«
gru_1/gru_cell_1/addAddV2gru_1/gru_cell_1/split:output:0!gru_1/gru_cell_1/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_1/gru_cell_1/add
gru_1/gru_cell_1/SigmoidSigmoidgru_1/gru_cell_1/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_1/gru_cell_1/Sigmoid¯
gru_1/gru_cell_1/add_1AddV2gru_1/gru_cell_1/split:output:1!gru_1/gru_cell_1/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_1/gru_cell_1/add_1
gru_1/gru_cell_1/Sigmoid_1Sigmoidgru_1/gru_cell_1/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_1/gru_cell_1/Sigmoid_1¨
gru_1/gru_cell_1/mulMulgru_1/gru_cell_1/Sigmoid_1:y:0!gru_1/gru_cell_1/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_1/gru_cell_1/mul¦
gru_1/gru_cell_1/add_2AddV2gru_1/gru_cell_1/split:output:2gru_1/gru_cell_1/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_1/gru_cell_1/add_2
gru_1/gru_cell_1/ReluRelugru_1/gru_cell_1/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_1/gru_cell_1/Relu
gru_1/gru_cell_1/mul_1Mulgru_1/gru_cell_1/Sigmoid:y:0gru_1/zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_1/gru_cell_1/mul_1u
gru_1/gru_cell_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru_1/gru_cell_1/sub/x¤
gru_1/gru_cell_1/subSubgru_1/gru_cell_1/sub/x:output:0gru_1/gru_cell_1/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_1/gru_cell_1/sub¨
gru_1/gru_cell_1/mul_2Mulgru_1/gru_cell_1/sub:z:0#gru_1/gru_cell_1/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_1/gru_cell_1/mul_2£
gru_1/gru_cell_1/add_3AddV2gru_1/gru_cell_1/mul_1:z:0gru_1/gru_cell_1/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_1/gru_cell_1/add_3
#gru_1/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2%
#gru_1/TensorArrayV2_1/element_shapeÐ
gru_1/TensorArrayV2_1TensorListReserve,gru_1/TensorArrayV2_1/element_shape:output:0gru_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
gru_1/TensorArrayV2_1Z

gru_1/timeConst*
_output_shapes
: *
dtype0*
value	B : 2

gru_1/time
gru_1/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2 
gru_1/while/maximum_iterationsv
gru_1/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
gru_1/while/loop_counter÷
gru_1/whileWhile!gru_1/while/loop_counter:output:0'gru_1/while/maximum_iterations:output:0gru_1/time:output:0gru_1/TensorArrayV2_1:handle:0gru_1/zeros:output:0gru_1/strided_slice_1:output:0=gru_1/TensorArrayUnstack/TensorListFromTensor:output_handle:0(gru_1_gru_cell_1_readvariableop_resource/gru_1_gru_cell_1_matmul_readvariableop_resource1gru_1_gru_cell_1_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	*!
bodyR
gru_1_while_body_9669*!
condR
gru_1_while_cond_9668*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 2
gru_1/whileÁ
6gru_1/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   28
6gru_1/TensorArrayV2Stack/TensorListStack/element_shape
(gru_1/TensorArrayV2Stack/TensorListStackTensorListStackgru_1/while:output:3?gru_1/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02*
(gru_1/TensorArrayV2Stack/TensorListStack
gru_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
gru_1/strided_slice_3/stack
gru_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
gru_1/strided_slice_3/stack_1
gru_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru_1/strided_slice_3/stack_2¾
gru_1/strided_slice_3StridedSlice1gru_1/TensorArrayV2Stack/TensorListStack:tensor:0$gru_1/strided_slice_3/stack:output:0&gru_1/strided_slice_3/stack_1:output:0&gru_1/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
gru_1/strided_slice_3
gru_1/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
gru_1/transpose_1/perm½
gru_1/transpose_1	Transpose1gru_1/TensorArrayV2Stack/TensorListStack:tensor:0gru_1/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_1/transpose_1r
gru_1/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
gru_1/runtime
dropout_1/IdentityIdentitygru_1/strided_slice_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout_1/Identity
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense/MatMul/ReadVariableOp
dense/MatMulMatMuldropout_1/Identity:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/MatMul
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
dense/BiasAdd/ReadVariableOp
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/BiasAdd¡
IdentityIdentitydense/BiasAdd:output:0^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp#^gru/gru_cell/MatMul/ReadVariableOp%^gru/gru_cell/MatMul_1/ReadVariableOp^gru/gru_cell/ReadVariableOp
^gru/while'^gru_1/gru_cell_1/MatMul/ReadVariableOp)^gru_1/gru_cell_1/MatMul_1/ReadVariableOp ^gru_1/gru_cell_1/ReadVariableOp^gru_1/while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:ÿÿÿÿÿÿÿÿÿ::::::::2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2H
"gru/gru_cell/MatMul/ReadVariableOp"gru/gru_cell/MatMul/ReadVariableOp2L
$gru/gru_cell/MatMul_1/ReadVariableOp$gru/gru_cell/MatMul_1/ReadVariableOp2:
gru/gru_cell/ReadVariableOpgru/gru_cell/ReadVariableOp2
	gru/while	gru/while2P
&gru_1/gru_cell_1/MatMul/ReadVariableOp&gru_1/gru_cell_1/MatMul/ReadVariableOp2T
(gru_1/gru_cell_1/MatMul_1/ReadVariableOp(gru_1/gru_cell_1/MatMul_1/ReadVariableOp2B
gru_1/gru_cell_1/ReadVariableOpgru_1/gru_cell_1/ReadVariableOp2
gru_1/whilegru_1/while:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


%__inference_gru_1_layer_call_fn_11184
inputs_0
unknown
	unknown_0
	unknown_1
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_gru_1_layer_call_and_return_conditional_losses_80472
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ :::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
inputs/0
Æ
 
while_cond_9876
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_12
.while_while_cond_9876___redundant_placeholder02
.while_while_cond_9876___redundant_placeholder12
.while_while_cond_9876___redundant_placeholder22
.while_while_cond_9876___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
:
	
Ø
?__inference_dense_layer_call_and_return_conditional_losses_8942

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¨
Ø
)__inference_sequential_layer_call_fn_9787

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity¢StatefulPartitionedCallÅ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_90122
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:ÿÿÿÿÿÿÿÿÿ::::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
­

#__inference_gru_layer_call_fn_10137
inputs_0
unknown
	unknown_0
	unknown_1
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *F
fAR?
=__inference_gru_layer_call_and_return_conditional_losses_74852
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
ÒX
Å
>__inference_gru_layer_call_and_return_conditional_losses_10466

inputs$
 gru_cell_readvariableop_resource+
'gru_cell_matmul_readvariableop_resource-
)gru_cell_matmul_1_readvariableop_resource
identity¢gru_cell/MatMul/ReadVariableOp¢ gru_cell/MatMul_1/ReadVariableOp¢gru_cell/ReadVariableOp¢whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ü
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_2
gru_cell/ReadVariableOpReadVariableOp gru_cell_readvariableop_resource*
_output_shapes

:`*
dtype02
gru_cell/ReadVariableOp
gru_cell/unstackUnpackgru_cell/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num2
gru_cell/unstack¨
gru_cell/MatMul/ReadVariableOpReadVariableOp'gru_cell_matmul_readvariableop_resource*
_output_shapes

:`*
dtype02 
gru_cell/MatMul/ReadVariableOp 
gru_cell/MatMulMatMulstrided_slice_2:output:0&gru_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
gru_cell/MatMul
gru_cell/BiasAddBiasAddgru_cell/MatMul:product:0gru_cell/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
gru_cell/BiasAddb
gru_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru_cell/Const
gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
gru_cell/split/split_dimÐ
gru_cell/splitSplit!gru_cell/split/split_dim:output:0gru_cell/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
gru_cell/split®
 gru_cell/MatMul_1/ReadVariableOpReadVariableOp)gru_cell_matmul_1_readvariableop_resource*
_output_shapes

: `*
dtype02"
 gru_cell/MatMul_1/ReadVariableOp
gru_cell/MatMul_1MatMulzeros:output:0(gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
gru_cell/MatMul_1
gru_cell/BiasAdd_1BiasAddgru_cell/MatMul_1:product:0gru_cell/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
gru_cell/BiasAdd_1y
gru_cell/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"        ÿÿÿÿ2
gru_cell/Const_1
gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
gru_cell/split_1/split_dim
gru_cell/split_1SplitVgru_cell/BiasAdd_1:output:0gru_cell/Const_1:output:0#gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
gru_cell/split_1
gru_cell/addAddV2gru_cell/split:output:0gru_cell/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/adds
gru_cell/SigmoidSigmoidgru_cell/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/Sigmoid
gru_cell/add_1AddV2gru_cell/split:output:1gru_cell/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/add_1y
gru_cell/Sigmoid_1Sigmoidgru_cell/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/Sigmoid_1
gru_cell/mulMulgru_cell/Sigmoid_1:y:0gru_cell/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/mul
gru_cell/add_2AddV2gru_cell/split:output:2gru_cell/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/add_2l
gru_cell/ReluRelugru_cell/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/Relu
gru_cell/mul_1Mulgru_cell/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/mul_1e
gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru_cell/sub/x
gru_cell/subSubgru_cell/sub/x:output:0gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/sub
gru_cell/mul_2Mulgru_cell/sub:z:0gru_cell/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/mul_2
gru_cell/add_3AddV2gru_cell/mul_1:z:0gru_cell/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/add_3
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2
TensorArrayV2_1/element_shape¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0 gru_cell_readvariableop_resource'gru_cell_matmul_readvariableop_resource)gru_cell_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ : : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_10376*
condR
while_cond_10375*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ : : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    22
0TensorArrayV2Stack/TensorListStack/element_shapeè
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¥
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeÍ
IdentityIdentitytranspose_1:y:0^gru_cell/MatMul/ReadVariableOp!^gru_cell/MatMul_1/ReadVariableOp^gru_cell/ReadVariableOp^while*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ:::2@
gru_cell/MatMul/ReadVariableOpgru_cell/MatMul/ReadVariableOp2D
 gru_cell/MatMul_1/ReadVariableOp gru_cell/MatMul_1/ReadVariableOp22
gru_cell/ReadVariableOpgru_cell/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¾
©
D__inference_gru_cell_1_layer_call_and_return_conditional_losses_7684

inputs

states
readvariableop_resource"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource
identity

identity_1¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp¢ReadVariableOpx
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:0*
dtype02
ReadVariableOpj
unstackUnpackReadVariableOp:value:0*
T0* 
_output_shapes
:0:0*	
num2	
unstack
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: 0*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02
MatMuls
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constm
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
split/split_dim¬
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
split
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:0*
dtype02
MatMul_1/ReadVariableOpy
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02

MatMul_1y
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02
	BiasAdd_1g
Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      ÿÿÿÿ2	
Const_1q
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
split_1/split_dimÓ
split_1SplitVBiasAdd_1:output:0Const_1:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2	
split_1g
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Sigmoidk
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	Sigmoid_1d
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mulb
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add_2Q
ReluRelu	add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu\
mul_1MulSigmoid:y:0states*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
sub/x`
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
subd
mul_2Mulsub:z:0Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mul_2_
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add_3 
IdentityIdentity	add_3:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity¤

Identity_1Identity	add_3:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*E
_input_shapes4
2:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ:::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namestates
º
¡
!__inference__traced_restore_11676
file_prefix!
assignvariableop_dense_kernel!
assignvariableop_1_dense_bias 
assignvariableop_2_adam_iter"
assignvariableop_3_adam_beta_1"
assignvariableop_4_adam_beta_2!
assignvariableop_5_adam_decay)
%assignvariableop_6_adam_learning_rate*
&assignvariableop_7_gru_gru_cell_kernel4
0assignvariableop_8_gru_gru_cell_recurrent_kernel(
$assignvariableop_9_gru_gru_cell_bias/
+assignvariableop_10_gru_1_gru_cell_1_kernel9
5assignvariableop_11_gru_1_gru_cell_1_recurrent_kernel-
)assignvariableop_12_gru_1_gru_cell_1_bias
assignvariableop_13_total
assignvariableop_14_count+
'assignvariableop_15_adam_dense_kernel_m)
%assignvariableop_16_adam_dense_bias_m2
.assignvariableop_17_adam_gru_gru_cell_kernel_m<
8assignvariableop_18_adam_gru_gru_cell_recurrent_kernel_m0
,assignvariableop_19_adam_gru_gru_cell_bias_m6
2assignvariableop_20_adam_gru_1_gru_cell_1_kernel_m@
<assignvariableop_21_adam_gru_1_gru_cell_1_recurrent_kernel_m4
0assignvariableop_22_adam_gru_1_gru_cell_1_bias_m+
'assignvariableop_23_adam_dense_kernel_v)
%assignvariableop_24_adam_dense_bias_v2
.assignvariableop_25_adam_gru_gru_cell_kernel_v<
8assignvariableop_26_adam_gru_gru_cell_recurrent_kernel_v0
,assignvariableop_27_adam_gru_gru_cell_bias_v6
2assignvariableop_28_adam_gru_1_gru_cell_1_kernel_v@
<assignvariableop_29_adam_gru_1_gru_cell_1_recurrent_kernel_v4
0assignvariableop_30_adam_gru_1_gru_cell_1_bias_v
identity_32¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9Ò
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
: *
dtype0*Þ
valueÔBÑ B6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesÎ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
: *
dtype0*S
valueJBH B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesÎ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
_output_shapes
::::::::::::::::::::::::::::::::*.
dtypes$
"2 	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity
AssignVariableOpAssignVariableOpassignvariableop_dense_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1¢
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_2¡
AssignVariableOp_2AssignVariableOpassignvariableop_2_adam_iterIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3£
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_beta_1Identity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4£
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_beta_2Identity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5¢
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_decayIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6ª
AssignVariableOp_6AssignVariableOp%assignvariableop_6_adam_learning_rateIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7«
AssignVariableOp_7AssignVariableOp&assignvariableop_7_gru_gru_cell_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8µ
AssignVariableOp_8AssignVariableOp0assignvariableop_8_gru_gru_cell_recurrent_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9©
AssignVariableOp_9AssignVariableOp$assignvariableop_9_gru_gru_cell_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10³
AssignVariableOp_10AssignVariableOp+assignvariableop_10_gru_1_gru_cell_1_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11½
AssignVariableOp_11AssignVariableOp5assignvariableop_11_gru_1_gru_cell_1_recurrent_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12±
AssignVariableOp_12AssignVariableOp)assignvariableop_12_gru_1_gru_cell_1_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13¡
AssignVariableOp_13AssignVariableOpassignvariableop_13_totalIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14¡
AssignVariableOp_14AssignVariableOpassignvariableop_14_countIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15¯
AssignVariableOp_15AssignVariableOp'assignvariableop_15_adam_dense_kernel_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16­
AssignVariableOp_16AssignVariableOp%assignvariableop_16_adam_dense_bias_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17¶
AssignVariableOp_17AssignVariableOp.assignvariableop_17_adam_gru_gru_cell_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18À
AssignVariableOp_18AssignVariableOp8assignvariableop_18_adam_gru_gru_cell_recurrent_kernel_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19´
AssignVariableOp_19AssignVariableOp,assignvariableop_19_adam_gru_gru_cell_bias_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20º
AssignVariableOp_20AssignVariableOp2assignvariableop_20_adam_gru_1_gru_cell_1_kernel_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21Ä
AssignVariableOp_21AssignVariableOp<assignvariableop_21_adam_gru_1_gru_cell_1_recurrent_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22¸
AssignVariableOp_22AssignVariableOp0assignvariableop_22_adam_gru_1_gru_cell_1_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23¯
AssignVariableOp_23AssignVariableOp'assignvariableop_23_adam_dense_kernel_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24­
AssignVariableOp_24AssignVariableOp%assignvariableop_24_adam_dense_bias_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25¶
AssignVariableOp_25AssignVariableOp.assignvariableop_25_adam_gru_gru_cell_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26À
AssignVariableOp_26AssignVariableOp8assignvariableop_26_adam_gru_gru_cell_recurrent_kernel_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27´
AssignVariableOp_27AssignVariableOp,assignvariableop_27_adam_gru_gru_cell_bias_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28º
AssignVariableOp_28AssignVariableOp2assignvariableop_28_adam_gru_1_gru_cell_1_kernel_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29Ä
AssignVariableOp_29AssignVariableOp<assignvariableop_29_adam_gru_1_gru_cell_1_recurrent_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30¸
AssignVariableOp_30AssignVariableOp0assignvariableop_30_adam_gru_1_gru_cell_1_bias_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_309
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp
Identity_31Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_31û
Identity_32IdentityIdentity_31:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_32"#
identity_32Identity_32:output:0*
_input_shapes
~: :::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
é
¾
D__inference_sequential_layer_call_and_return_conditional_losses_8984
	gru_input
gru_8962
gru_8964
gru_8966

gru_1_8970

gru_1_8972

gru_1_8974

dense_8978

dense_8980
identity¢dense/StatefulPartitionedCall¢gru/StatefulPartitionedCall¢gru_1/StatefulPartitionedCall
gru/StatefulPartitionedCallStatefulPartitionedCall	gru_inputgru_8962gru_8964gru_8966*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *F
fAR?
=__inference_gru_layer_call_and_return_conditional_losses_84952
gru/StatefulPartitionedCallô
dropout/PartitionedCallPartitionedCall$gru/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_85422
dropout/PartitionedCallª
gru_1/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0
gru_1_8970
gru_1_8972
gru_1_8974*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_gru_1_layer_call_and_return_conditional_losses_88722
gru_1/StatefulPartitionedCallø
dropout_1/PartitionedCallPartitionedCall&gru_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dropout_1_layer_call_and_return_conditional_losses_89192
dropout_1/PartitionedCall
dense/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0
dense_8978
dense_8980*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_89422
dense/StatefulPartitionedCallØ
IdentityIdentity&dense/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall^gru/StatefulPartitionedCall^gru_1/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:ÿÿÿÿÿÿÿÿÿ::::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2:
gru/StatefulPartitionedCallgru/StatefulPartitionedCall2>
gru_1/StatefulPartitionedCallgru_1/StatefulPartitionedCall:V R
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#
_user_specified_name	gru_input
Ç
¬
E__inference_gru_cell_1_layer_call_and_return_conditional_losses_11429

inputs
states_0
readvariableop_resource"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource
identity

identity_1¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp¢ReadVariableOpx
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:0*
dtype02
ReadVariableOpj
unstackUnpackReadVariableOp:value:0*
T0* 
_output_shapes
:0:0*	
num2	
unstack
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: 0*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02
MatMuls
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constm
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
split/split_dim¬
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
split
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:0*
dtype02
MatMul_1/ReadVariableOp{
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02

MatMul_1y
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02
	BiasAdd_1g
Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      ÿÿÿÿ2	
Const_1q
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
split_1/split_dimÓ
split_1SplitVBiasAdd_1:output:0Const_1:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2	
split_1g
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Sigmoidk
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	Sigmoid_1d
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mulb
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add_2Q
ReluRelu	add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu^
mul_1MulSigmoid:y:0states_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
sub/x`
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
subd
mul_2Mulsub:z:0Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mul_2_
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add_3 
IdentityIdentity	add_3:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity¤

Identity_1Identity	add_3:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*E
_input_shapes4
2:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ:::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/0
]
Ó

sequential_gru_while_body_6797:
6sequential_gru_while_sequential_gru_while_loop_counter@
<sequential_gru_while_sequential_gru_while_maximum_iterations$
 sequential_gru_while_placeholder&
"sequential_gru_while_placeholder_1&
"sequential_gru_while_placeholder_29
5sequential_gru_while_sequential_gru_strided_slice_1_0u
qsequential_gru_while_tensorarrayv2read_tensorlistgetitem_sequential_gru_tensorarrayunstack_tensorlistfromtensor_0;
7sequential_gru_while_gru_cell_readvariableop_resource_0B
>sequential_gru_while_gru_cell_matmul_readvariableop_resource_0D
@sequential_gru_while_gru_cell_matmul_1_readvariableop_resource_0!
sequential_gru_while_identity#
sequential_gru_while_identity_1#
sequential_gru_while_identity_2#
sequential_gru_while_identity_3#
sequential_gru_while_identity_47
3sequential_gru_while_sequential_gru_strided_slice_1s
osequential_gru_while_tensorarrayv2read_tensorlistgetitem_sequential_gru_tensorarrayunstack_tensorlistfromtensor9
5sequential_gru_while_gru_cell_readvariableop_resource@
<sequential_gru_while_gru_cell_matmul_readvariableop_resourceB
>sequential_gru_while_gru_cell_matmul_1_readvariableop_resource¢3sequential/gru/while/gru_cell/MatMul/ReadVariableOp¢5sequential/gru/while/gru_cell/MatMul_1/ReadVariableOp¢,sequential/gru/while/gru_cell/ReadVariableOpá
Fsequential/gru/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2H
Fsequential/gru/while/TensorArrayV2Read/TensorListGetItem/element_shape­
8sequential/gru/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemqsequential_gru_while_tensorarrayv2read_tensorlistgetitem_sequential_gru_tensorarrayunstack_tensorlistfromtensor_0 sequential_gru_while_placeholderOsequential/gru/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02:
8sequential/gru/while/TensorArrayV2Read/TensorListGetItemÔ
,sequential/gru/while/gru_cell/ReadVariableOpReadVariableOp7sequential_gru_while_gru_cell_readvariableop_resource_0*
_output_shapes

:`*
dtype02.
,sequential/gru/while/gru_cell/ReadVariableOpÄ
%sequential/gru/while/gru_cell/unstackUnpack4sequential/gru/while/gru_cell/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num2'
%sequential/gru/while/gru_cell/unstacké
3sequential/gru/while/gru_cell/MatMul/ReadVariableOpReadVariableOp>sequential_gru_while_gru_cell_matmul_readvariableop_resource_0*
_output_shapes

:`*
dtype025
3sequential/gru/while/gru_cell/MatMul/ReadVariableOp
$sequential/gru/while/gru_cell/MatMulMatMul?sequential/gru/while/TensorArrayV2Read/TensorListGetItem:item:0;sequential/gru/while/gru_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2&
$sequential/gru/while/gru_cell/MatMulë
%sequential/gru/while/gru_cell/BiasAddBiasAdd.sequential/gru/while/gru_cell/MatMul:product:0.sequential/gru/while/gru_cell/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2'
%sequential/gru/while/gru_cell/BiasAdd
#sequential/gru/while/gru_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2%
#sequential/gru/while/gru_cell/Const©
-sequential/gru/while/gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2/
-sequential/gru/while/gru_cell/split/split_dim¤
#sequential/gru/while/gru_cell/splitSplit6sequential/gru/while/gru_cell/split/split_dim:output:0.sequential/gru/while/gru_cell/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2%
#sequential/gru/while/gru_cell/splitï
5sequential/gru/while/gru_cell/MatMul_1/ReadVariableOpReadVariableOp@sequential_gru_while_gru_cell_matmul_1_readvariableop_resource_0*
_output_shapes

: `*
dtype027
5sequential/gru/while/gru_cell/MatMul_1/ReadVariableOpï
&sequential/gru/while/gru_cell/MatMul_1MatMul"sequential_gru_while_placeholder_2=sequential/gru/while/gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2(
&sequential/gru/while/gru_cell/MatMul_1ñ
'sequential/gru/while/gru_cell/BiasAdd_1BiasAdd0sequential/gru/while/gru_cell/MatMul_1:product:0.sequential/gru/while/gru_cell/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2)
'sequential/gru/while/gru_cell/BiasAdd_1£
%sequential/gru/while/gru_cell/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"        ÿÿÿÿ2'
%sequential/gru/while/gru_cell/Const_1­
/sequential/gru/while/gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ21
/sequential/gru/while/gru_cell/split_1/split_dimé
%sequential/gru/while/gru_cell/split_1SplitV0sequential/gru/while/gru_cell/BiasAdd_1:output:0.sequential/gru/while/gru_cell/Const_1:output:08sequential/gru/while/gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2'
%sequential/gru/while/gru_cell/split_1ß
!sequential/gru/while/gru_cell/addAddV2,sequential/gru/while/gru_cell/split:output:0.sequential/gru/while/gru_cell/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!sequential/gru/while/gru_cell/add²
%sequential/gru/while/gru_cell/SigmoidSigmoid%sequential/gru/while/gru_cell/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2'
%sequential/gru/while/gru_cell/Sigmoidã
#sequential/gru/while/gru_cell/add_1AddV2,sequential/gru/while/gru_cell/split:output:1.sequential/gru/while/gru_cell/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2%
#sequential/gru/while/gru_cell/add_1¸
'sequential/gru/while/gru_cell/Sigmoid_1Sigmoid'sequential/gru/while/gru_cell/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2)
'sequential/gru/while/gru_cell/Sigmoid_1Ü
!sequential/gru/while/gru_cell/mulMul+sequential/gru/while/gru_cell/Sigmoid_1:y:0.sequential/gru/while/gru_cell/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!sequential/gru/while/gru_cell/mulÚ
#sequential/gru/while/gru_cell/add_2AddV2,sequential/gru/while/gru_cell/split:output:2%sequential/gru/while/gru_cell/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2%
#sequential/gru/while/gru_cell/add_2«
"sequential/gru/while/gru_cell/ReluRelu'sequential/gru/while/gru_cell/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2$
"sequential/gru/while/gru_cell/ReluÒ
#sequential/gru/while/gru_cell/mul_1Mul)sequential/gru/while/gru_cell/Sigmoid:y:0"sequential_gru_while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2%
#sequential/gru/while/gru_cell/mul_1
#sequential/gru/while/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2%
#sequential/gru/while/gru_cell/sub/xØ
!sequential/gru/while/gru_cell/subSub,sequential/gru/while/gru_cell/sub/x:output:0)sequential/gru/while/gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!sequential/gru/while/gru_cell/subÜ
#sequential/gru/while/gru_cell/mul_2Mul%sequential/gru/while/gru_cell/sub:z:00sequential/gru/while/gru_cell/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2%
#sequential/gru/while/gru_cell/mul_2×
#sequential/gru/while/gru_cell/add_3AddV2'sequential/gru/while/gru_cell/mul_1:z:0'sequential/gru/while/gru_cell/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2%
#sequential/gru/while/gru_cell/add_3§
9sequential/gru/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem"sequential_gru_while_placeholder_1 sequential_gru_while_placeholder'sequential/gru/while/gru_cell/add_3:z:0*
_output_shapes
: *
element_dtype02;
9sequential/gru/while/TensorArrayV2Write/TensorListSetItemz
sequential/gru/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
sequential/gru/while/add/y¥
sequential/gru/while/addAddV2 sequential_gru_while_placeholder#sequential/gru/while/add/y:output:0*
T0*
_output_shapes
: 2
sequential/gru/while/add~
sequential/gru/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
sequential/gru/while/add_1/yÁ
sequential/gru/while/add_1AddV26sequential_gru_while_sequential_gru_while_loop_counter%sequential/gru/while/add_1/y:output:0*
T0*
_output_shapes
: 2
sequential/gru/while/add_1¨
sequential/gru/while/IdentityIdentitysequential/gru/while/add_1:z:04^sequential/gru/while/gru_cell/MatMul/ReadVariableOp6^sequential/gru/while/gru_cell/MatMul_1/ReadVariableOp-^sequential/gru/while/gru_cell/ReadVariableOp*
T0*
_output_shapes
: 2
sequential/gru/while/IdentityÊ
sequential/gru/while/Identity_1Identity<sequential_gru_while_sequential_gru_while_maximum_iterations4^sequential/gru/while/gru_cell/MatMul/ReadVariableOp6^sequential/gru/while/gru_cell/MatMul_1/ReadVariableOp-^sequential/gru/while/gru_cell/ReadVariableOp*
T0*
_output_shapes
: 2!
sequential/gru/while/Identity_1ª
sequential/gru/while/Identity_2Identitysequential/gru/while/add:z:04^sequential/gru/while/gru_cell/MatMul/ReadVariableOp6^sequential/gru/while/gru_cell/MatMul_1/ReadVariableOp-^sequential/gru/while/gru_cell/ReadVariableOp*
T0*
_output_shapes
: 2!
sequential/gru/while/Identity_2×
sequential/gru/while/Identity_3IdentityIsequential/gru/while/TensorArrayV2Write/TensorListSetItem:output_handle:04^sequential/gru/while/gru_cell/MatMul/ReadVariableOp6^sequential/gru/while/gru_cell/MatMul_1/ReadVariableOp-^sequential/gru/while/gru_cell/ReadVariableOp*
T0*
_output_shapes
: 2!
sequential/gru/while/Identity_3Æ
sequential/gru/while/Identity_4Identity'sequential/gru/while/gru_cell/add_3:z:04^sequential/gru/while/gru_cell/MatMul/ReadVariableOp6^sequential/gru/while/gru_cell/MatMul_1/ReadVariableOp-^sequential/gru/while/gru_cell/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
sequential/gru/while/Identity_4"
>sequential_gru_while_gru_cell_matmul_1_readvariableop_resource@sequential_gru_while_gru_cell_matmul_1_readvariableop_resource_0"~
<sequential_gru_while_gru_cell_matmul_readvariableop_resource>sequential_gru_while_gru_cell_matmul_readvariableop_resource_0"p
5sequential_gru_while_gru_cell_readvariableop_resource7sequential_gru_while_gru_cell_readvariableop_resource_0"G
sequential_gru_while_identity&sequential/gru/while/Identity:output:0"K
sequential_gru_while_identity_1(sequential/gru/while/Identity_1:output:0"K
sequential_gru_while_identity_2(sequential/gru/while/Identity_2:output:0"K
sequential_gru_while_identity_3(sequential/gru/while/Identity_3:output:0"K
sequential_gru_while_identity_4(sequential/gru/while/Identity_4:output:0"l
3sequential_gru_while_sequential_gru_strided_slice_15sequential_gru_while_sequential_gru_strided_slice_1_0"ä
osequential_gru_while_tensorarrayv2read_tensorlistgetitem_sequential_gru_tensorarrayunstack_tensorlistfromtensorqsequential_gru_while_tensorarrayv2read_tensorlistgetitem_sequential_gru_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :ÿÿÿÿÿÿÿÿÿ : : :::2j
3sequential/gru/while/gru_cell/MatMul/ReadVariableOp3sequential/gru/while/gru_cell/MatMul/ReadVariableOp2n
5sequential/gru/while/gru_cell/MatMul_1/ReadVariableOp5sequential/gru/while/gru_cell/MatMul_1/ReadVariableOp2\
,sequential/gru/while/gru_cell/ReadVariableOp,sequential/gru/while/gru_cell/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
: 


#__inference_gru_layer_call_fn_10488

inputs
unknown
	unknown_0
	unknown_1
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *F
fAR?
=__inference_gru_layer_call_and_return_conditional_losses_84952
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ:::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Å
ª
C__inference_gru_cell_layer_call_and_return_conditional_losses_11281

inputs
states_0
readvariableop_resource"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource
identity

identity_1¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp¢ReadVariableOpx
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:`*
dtype02
ReadVariableOpj
unstackUnpackReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num2	
unstack
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:`*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
MatMuls
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constm
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
split/split_dim¬
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
split
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

: `*
dtype02
MatMul_1/ReadVariableOp{
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2

MatMul_1y
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
	BiasAdd_1g
Const_1Const*
_output_shapes
:*
dtype0*!
valueB"        ÿÿÿÿ2	
Const_1q
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
split_1/split_dimÓ
split_1SplitVBiasAdd_1:output:0Const_1:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2	
split_1g
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2	
Sigmoidk
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
	Sigmoid_1d
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
mulb
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
add_2Q
ReluRelu	add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Relu^
mul_1MulSigmoid:y:0states_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
sub/x`
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
subd
mul_2Mulsub:z:0Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
mul_2_
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
add_3 
IdentityIdentity	add_3:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity¤

Identity_1Identity	add_3:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*E
_input_shapes4
2:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ :::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
states/0
Æ
 
while_cond_8622
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_12
.while_while_cond_8622___redundant_placeholder02
.while_while_cond_8622___redundant_placeholder12
.while_while_cond_8622___redundant_placeholder22
.while_while_cond_8622___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
µ
ã
 sequential_gru_1_while_cond_6952>
:sequential_gru_1_while_sequential_gru_1_while_loop_counterD
@sequential_gru_1_while_sequential_gru_1_while_maximum_iterations&
"sequential_gru_1_while_placeholder(
$sequential_gru_1_while_placeholder_1(
$sequential_gru_1_while_placeholder_2@
<sequential_gru_1_while_less_sequential_gru_1_strided_slice_1T
Psequential_gru_1_while_sequential_gru_1_while_cond_6952___redundant_placeholder0T
Psequential_gru_1_while_sequential_gru_1_while_cond_6952___redundant_placeholder1T
Psequential_gru_1_while_sequential_gru_1_while_cond_6952___redundant_placeholder2T
Psequential_gru_1_while_sequential_gru_1_while_cond_6952___redundant_placeholder3#
sequential_gru_1_while_identity
Å
sequential/gru_1/while/LessLess"sequential_gru_1_while_placeholder<sequential_gru_1_while_less_sequential_gru_1_strided_slice_1*
T0*
_output_shapes
: 2
sequential/gru_1/while/Less
sequential/gru_1/while/IdentityIdentitysequential/gru_1/while/Less:z:0*
T0
*
_output_shapes
: 2!
sequential/gru_1/while/Identity"K
sequential_gru_1_while_identity(sequential/gru_1/while/Identity:output:0*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
Æ
 
while_cond_7420
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_12
.while_while_cond_7420___redundant_placeholder02
.while_while_cond_7420___redundant_placeholder12
.while_while_cond_7420___redundant_placeholder22
.while_while_cond_7420___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
:
õ

gru_1_while_cond_9668(
$gru_1_while_gru_1_while_loop_counter.
*gru_1_while_gru_1_while_maximum_iterations
gru_1_while_placeholder
gru_1_while_placeholder_1
gru_1_while_placeholder_2*
&gru_1_while_less_gru_1_strided_slice_1>
:gru_1_while_gru_1_while_cond_9668___redundant_placeholder0>
:gru_1_while_gru_1_while_cond_9668___redundant_placeholder1>
:gru_1_while_gru_1_while_cond_9668___redundant_placeholder2>
:gru_1_while_gru_1_while_cond_9668___redundant_placeholder3
gru_1_while_identity

gru_1/while/LessLessgru_1_while_placeholder&gru_1_while_less_gru_1_strided_slice_1*
T0*
_output_shapes
: 2
gru_1/while/Lesso
gru_1/while/IdentityIdentitygru_1/while/Less:z:0*
T0
*
_output_shapes
: 2
gru_1/while/Identity"5
gru_1_while_identitygru_1/while/Identity:output:0*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
±K

gru_while_body_9513$
 gru_while_gru_while_loop_counter*
&gru_while_gru_while_maximum_iterations
gru_while_placeholder
gru_while_placeholder_1
gru_while_placeholder_2#
gru_while_gru_strided_slice_1_0_
[gru_while_tensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensor_00
,gru_while_gru_cell_readvariableop_resource_07
3gru_while_gru_cell_matmul_readvariableop_resource_09
5gru_while_gru_cell_matmul_1_readvariableop_resource_0
gru_while_identity
gru_while_identity_1
gru_while_identity_2
gru_while_identity_3
gru_while_identity_4!
gru_while_gru_strided_slice_1]
Ygru_while_tensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensor.
*gru_while_gru_cell_readvariableop_resource5
1gru_while_gru_cell_matmul_readvariableop_resource7
3gru_while_gru_cell_matmul_1_readvariableop_resource¢(gru/while/gru_cell/MatMul/ReadVariableOp¢*gru/while/gru_cell/MatMul_1/ReadVariableOp¢!gru/while/gru_cell/ReadVariableOpË
;gru/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2=
;gru/while/TensorArrayV2Read/TensorListGetItem/element_shapeë
-gru/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem[gru_while_tensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensor_0gru_while_placeholderDgru/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02/
-gru/while/TensorArrayV2Read/TensorListGetItem³
!gru/while/gru_cell/ReadVariableOpReadVariableOp,gru_while_gru_cell_readvariableop_resource_0*
_output_shapes

:`*
dtype02#
!gru/while/gru_cell/ReadVariableOp£
gru/while/gru_cell/unstackUnpack)gru/while/gru_cell/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num2
gru/while/gru_cell/unstackÈ
(gru/while/gru_cell/MatMul/ReadVariableOpReadVariableOp3gru_while_gru_cell_matmul_readvariableop_resource_0*
_output_shapes

:`*
dtype02*
(gru/while/gru_cell/MatMul/ReadVariableOpÚ
gru/while/gru_cell/MatMulMatMul4gru/while/TensorArrayV2Read/TensorListGetItem:item:00gru/while/gru_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
gru/while/gru_cell/MatMul¿
gru/while/gru_cell/BiasAddBiasAdd#gru/while/gru_cell/MatMul:product:0#gru/while/gru_cell/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
gru/while/gru_cell/BiasAddv
gru/while/gru_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru/while/gru_cell/Const
"gru/while/gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2$
"gru/while/gru_cell/split/split_dimø
gru/while/gru_cell/splitSplit+gru/while/gru_cell/split/split_dim:output:0#gru/while/gru_cell/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
gru/while/gru_cell/splitÎ
*gru/while/gru_cell/MatMul_1/ReadVariableOpReadVariableOp5gru_while_gru_cell_matmul_1_readvariableop_resource_0*
_output_shapes

: `*
dtype02,
*gru/while/gru_cell/MatMul_1/ReadVariableOpÃ
gru/while/gru_cell/MatMul_1MatMulgru_while_placeholder_22gru/while/gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
gru/while/gru_cell/MatMul_1Å
gru/while/gru_cell/BiasAdd_1BiasAdd%gru/while/gru_cell/MatMul_1:product:0#gru/while/gru_cell/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
gru/while/gru_cell/BiasAdd_1
gru/while/gru_cell/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"        ÿÿÿÿ2
gru/while/gru_cell/Const_1
$gru/while/gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2&
$gru/while/gru_cell/split_1/split_dim²
gru/while/gru_cell/split_1SplitV%gru/while/gru_cell/BiasAdd_1:output:0#gru/while/gru_cell/Const_1:output:0-gru/while/gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
gru/while/gru_cell/split_1³
gru/while/gru_cell/addAddV2!gru/while/gru_cell/split:output:0#gru/while/gru_cell/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/while/gru_cell/add
gru/while/gru_cell/SigmoidSigmoidgru/while/gru_cell/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/while/gru_cell/Sigmoid·
gru/while/gru_cell/add_1AddV2!gru/while/gru_cell/split:output:1#gru/while/gru_cell/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/while/gru_cell/add_1
gru/while/gru_cell/Sigmoid_1Sigmoidgru/while/gru_cell/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/while/gru_cell/Sigmoid_1°
gru/while/gru_cell/mulMul gru/while/gru_cell/Sigmoid_1:y:0#gru/while/gru_cell/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/while/gru_cell/mul®
gru/while/gru_cell/add_2AddV2!gru/while/gru_cell/split:output:2gru/while/gru_cell/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/while/gru_cell/add_2
gru/while/gru_cell/ReluRelugru/while/gru_cell/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/while/gru_cell/Relu¦
gru/while/gru_cell/mul_1Mulgru/while/gru_cell/Sigmoid:y:0gru_while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/while/gru_cell/mul_1y
gru/while/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru/while/gru_cell/sub/x¬
gru/while/gru_cell/subSub!gru/while/gru_cell/sub/x:output:0gru/while/gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/while/gru_cell/sub°
gru/while/gru_cell/mul_2Mulgru/while/gru_cell/sub:z:0%gru/while/gru_cell/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/while/gru_cell/mul_2«
gru/while/gru_cell/add_3AddV2gru/while/gru_cell/mul_1:z:0gru/while/gru_cell/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/while/gru_cell/add_3ð
.gru/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemgru_while_placeholder_1gru_while_placeholdergru/while/gru_cell/add_3:z:0*
_output_shapes
: *
element_dtype020
.gru/while/TensorArrayV2Write/TensorListSetItemd
gru/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
gru/while/add/yy
gru/while/addAddV2gru_while_placeholdergru/while/add/y:output:0*
T0*
_output_shapes
: 2
gru/while/addh
gru/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
gru/while/add_1/y
gru/while/add_1AddV2 gru_while_gru_while_loop_countergru/while/add_1/y:output:0*
T0*
_output_shapes
: 2
gru/while/add_1æ
gru/while/IdentityIdentitygru/while/add_1:z:0)^gru/while/gru_cell/MatMul/ReadVariableOp+^gru/while/gru_cell/MatMul_1/ReadVariableOp"^gru/while/gru_cell/ReadVariableOp*
T0*
_output_shapes
: 2
gru/while/Identityý
gru/while/Identity_1Identity&gru_while_gru_while_maximum_iterations)^gru/while/gru_cell/MatMul/ReadVariableOp+^gru/while/gru_cell/MatMul_1/ReadVariableOp"^gru/while/gru_cell/ReadVariableOp*
T0*
_output_shapes
: 2
gru/while/Identity_1è
gru/while/Identity_2Identitygru/while/add:z:0)^gru/while/gru_cell/MatMul/ReadVariableOp+^gru/while/gru_cell/MatMul_1/ReadVariableOp"^gru/while/gru_cell/ReadVariableOp*
T0*
_output_shapes
: 2
gru/while/Identity_2
gru/while/Identity_3Identity>gru/while/TensorArrayV2Write/TensorListSetItem:output_handle:0)^gru/while/gru_cell/MatMul/ReadVariableOp+^gru/while/gru_cell/MatMul_1/ReadVariableOp"^gru/while/gru_cell/ReadVariableOp*
T0*
_output_shapes
: 2
gru/while/Identity_3
gru/while/Identity_4Identitygru/while/gru_cell/add_3:z:0)^gru/while/gru_cell/MatMul/ReadVariableOp+^gru/while/gru_cell/MatMul_1/ReadVariableOp"^gru/while/gru_cell/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/while/Identity_4"l
3gru_while_gru_cell_matmul_1_readvariableop_resource5gru_while_gru_cell_matmul_1_readvariableop_resource_0"h
1gru_while_gru_cell_matmul_readvariableop_resource3gru_while_gru_cell_matmul_readvariableop_resource_0"Z
*gru_while_gru_cell_readvariableop_resource,gru_while_gru_cell_readvariableop_resource_0"@
gru_while_gru_strided_slice_1gru_while_gru_strided_slice_1_0"1
gru_while_identitygru/while/Identity:output:0"5
gru_while_identity_1gru/while/Identity_1:output:0"5
gru_while_identity_2gru/while/Identity_2:output:0"5
gru_while_identity_3gru/while/Identity_3:output:0"5
gru_while_identity_4gru/while/Identity_4:output:0"¸
Ygru_while_tensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensor[gru_while_tensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :ÿÿÿÿÿÿÿÿÿ : : :::2T
(gru/while/gru_cell/MatMul/ReadVariableOp(gru/while/gru_cell/MatMul/ReadVariableOp2X
*gru/while/gru_cell/MatMul_1/ReadVariableOp*gru/while/gru_cell/MatMul_1/ReadVariableOp2F
!gru/while/gru_cell/ReadVariableOp!gru/while/gru_cell/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
: 
±
Û
)__inference_sequential_layer_call_fn_9077
	gru_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity¢StatefulPartitionedCallÈ
StatefulPartitionedCallStatefulPartitionedCall	gru_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_90582
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:ÿÿÿÿÿÿÿÿÿ::::::::22
StatefulPartitionedCallStatefulPartitionedCall:V R
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#
_user_specified_name	gru_input
¼
§
B__inference_gru_cell_layer_call_and_return_conditional_losses_7122

inputs

states
readvariableop_resource"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource
identity

identity_1¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp¢ReadVariableOpx
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:`*
dtype02
ReadVariableOpj
unstackUnpackReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num2	
unstack
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:`*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
MatMuls
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constm
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
split/split_dim¬
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
split
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

: `*
dtype02
MatMul_1/ReadVariableOpy
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2

MatMul_1y
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
	BiasAdd_1g
Const_1Const*
_output_shapes
:*
dtype0*!
valueB"        ÿÿÿÿ2	
Const_1q
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
split_1/split_dimÓ
split_1SplitVBiasAdd_1:output:0Const_1:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2	
split_1g
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2	
Sigmoidk
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
	Sigmoid_1d
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
mulb
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
add_2Q
ReluRelu	add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Relu\
mul_1MulSigmoid:y:0states*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
sub/x`
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
subd
mul_2Mulsub:z:0Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
mul_2_
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
add_3 
IdentityIdentity	add_3:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity¤

Identity_1Identity	add_3:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*E
_input_shapes4
2:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ :::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_namestates
±
Û
)__inference_sequential_layer_call_fn_9031
	gru_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity¢StatefulPartitionedCallÈ
StatefulPartitionedCallStatefulPartitionedCall	gru_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_90122
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:ÿÿÿÿÿÿÿÿÿ::::::::22
StatefulPartitionedCallStatefulPartitionedCall:V R
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#
_user_specified_name	gru_input
³!
Æ
while_body_7983
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_gru_cell_1_8005_0
while_gru_cell_1_8007_0
while_gru_cell_1_8009_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_gru_cell_1_8005
while_gru_cell_1_8007
while_gru_cell_1_8009¢(while/gru_cell_1/StatefulPartitionedCallÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÓ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem¦
(while/gru_cell_1/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_gru_cell_1_8005_0while_gru_cell_1_8007_0while_gru_cell_1_8009_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_gru_cell_1_layer_call_and_return_conditional_losses_76842*
(while/gru_cell_1/StatefulPartitionedCallõ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder1while/gru_cell_1/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1
while/IdentityIdentitywhile/add_1:z:0)^while/gru_cell_1/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity
while/Identity_1Identitywhile_while_maximum_iterations)^while/gru_cell_1/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1
while/Identity_2Identitywhile/add:z:0)^while/gru_cell_1/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2¸
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0)^while/gru_cell_1/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3À
while/Identity_4Identity1while/gru_cell_1/StatefulPartitionedCall:output:1)^while/gru_cell_1/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_4"0
while_gru_cell_1_8005while_gru_cell_1_8005_0"0
while_gru_cell_1_8007while_gru_cell_1_8007_0"0
while_gru_cell_1_8009while_gru_cell_1_8009_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :ÿÿÿÿÿÿÿÿÿ: : :::2T
(while/gru_cell_1/StatefulPartitionedCall(while/gru_cell_1/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
Ø	
ª
(__inference_gru_cell_layer_call_fn_11349

inputs
states_0
unknown
	unknown_0
	unknown_1
identity

identity_1¢StatefulPartitionedCall¡
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_gru_cell_layer_call_and_return_conditional_losses_71622
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*E
_input_shapes4
2:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ :::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
states/0
û 
¸
while_body_7539
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_gru_cell_7561_0
while_gru_cell_7563_0
while_gru_cell_7565_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_gru_cell_7561
while_gru_cell_7563
while_gru_cell_7565¢&while/gru_cell/StatefulPartitionedCallÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÓ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem
&while/gru_cell/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_gru_cell_7561_0while_gru_cell_7563_0while_gru_cell_7565_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_gru_cell_layer_call_and_return_conditional_losses_71622(
&while/gru_cell/StatefulPartitionedCalló
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder/while/gru_cell/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1
while/IdentityIdentitywhile/add_1:z:0'^while/gru_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity
while/Identity_1Identitywhile_while_maximum_iterations'^while/gru_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1
while/Identity_2Identitywhile/add:z:0'^while/gru_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2¶
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0'^while/gru_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3¼
while/Identity_4Identity/while/gru_cell/StatefulPartitionedCall:output:1'^while/gru_cell/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4",
while_gru_cell_7561while_gru_cell_7561_0",
while_gru_cell_7563while_gru_cell_7563_0",
while_gru_cell_7565while_gru_cell_7565_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :ÿÿÿÿÿÿÿÿÿ : : :::2P
&while/gru_cell/StatefulPartitionedCall&while/gru_cell/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
: 
Ë
¥
while_cond_10375
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_10375___redundant_placeholder03
/while_while_cond_10375___redundant_placeholder13
/while_while_cond_10375___redundant_placeholder23
/while_while_cond_10375___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
:
û 
¸
while_body_7421
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_gru_cell_7443_0
while_gru_cell_7445_0
while_gru_cell_7447_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_gru_cell_7443
while_gru_cell_7445
while_gru_cell_7447¢&while/gru_cell/StatefulPartitionedCallÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÓ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem
&while/gru_cell/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_gru_cell_7443_0while_gru_cell_7445_0while_gru_cell_7447_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_gru_cell_layer_call_and_return_conditional_losses_71222(
&while/gru_cell/StatefulPartitionedCalló
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder/while/gru_cell/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1
while/IdentityIdentitywhile/add_1:z:0'^while/gru_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity
while/Identity_1Identitywhile_while_maximum_iterations'^while/gru_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1
while/Identity_2Identitywhile/add:z:0'^while/gru_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2¶
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0'^while/gru_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3¼
while/Identity_4Identity/while/gru_cell/StatefulPartitionedCall:output:1'^while/gru_cell/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4",
while_gru_cell_7443while_gru_cell_7443_0",
while_gru_cell_7445while_gru_cell_7445_0",
while_gru_cell_7447while_gru_cell_7447_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :ÿÿÿÿÿÿÿÿÿ : : :::2P
&while/gru_cell/StatefulPartitionedCall&while/gru_cell/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
: 
·ù
»
__inference__wrapped_model_7050
	gru_input3
/sequential_gru_gru_cell_readvariableop_resource:
6sequential_gru_gru_cell_matmul_readvariableop_resource<
8sequential_gru_gru_cell_matmul_1_readvariableop_resource7
3sequential_gru_1_gru_cell_1_readvariableop_resource>
:sequential_gru_1_gru_cell_1_matmul_readvariableop_resource@
<sequential_gru_1_gru_cell_1_matmul_1_readvariableop_resource3
/sequential_dense_matmul_readvariableop_resource4
0sequential_dense_biasadd_readvariableop_resource
identity¢'sequential/dense/BiasAdd/ReadVariableOp¢&sequential/dense/MatMul/ReadVariableOp¢-sequential/gru/gru_cell/MatMul/ReadVariableOp¢/sequential/gru/gru_cell/MatMul_1/ReadVariableOp¢&sequential/gru/gru_cell/ReadVariableOp¢sequential/gru/while¢1sequential/gru_1/gru_cell_1/MatMul/ReadVariableOp¢3sequential/gru_1/gru_cell_1/MatMul_1/ReadVariableOp¢*sequential/gru_1/gru_cell_1/ReadVariableOp¢sequential/gru_1/whilee
sequential/gru/ShapeShape	gru_input*
T0*
_output_shapes
:2
sequential/gru/Shape
"sequential/gru/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"sequential/gru/strided_slice/stack
$sequential/gru/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$sequential/gru/strided_slice/stack_1
$sequential/gru/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$sequential/gru/strided_slice/stack_2¼
sequential/gru/strided_sliceStridedSlicesequential/gru/Shape:output:0+sequential/gru/strided_slice/stack:output:0-sequential/gru/strided_slice/stack_1:output:0-sequential/gru/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
sequential/gru/strided_slicez
sequential/gru/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
sequential/gru/zeros/mul/y¨
sequential/gru/zeros/mulMul%sequential/gru/strided_slice:output:0#sequential/gru/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
sequential/gru/zeros/mul}
sequential/gru/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
sequential/gru/zeros/Less/y£
sequential/gru/zeros/LessLesssequential/gru/zeros/mul:z:0$sequential/gru/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
sequential/gru/zeros/Less
sequential/gru/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
sequential/gru/zeros/packed/1¿
sequential/gru/zeros/packedPack%sequential/gru/strided_slice:output:0&sequential/gru/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
sequential/gru/zeros/packed}
sequential/gru/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
sequential/gru/zeros/Const±
sequential/gru/zerosFill$sequential/gru/zeros/packed:output:0#sequential/gru/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
sequential/gru/zeros
sequential/gru/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
sequential/gru/transpose/permª
sequential/gru/transpose	Transpose	gru_input&sequential/gru/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential/gru/transpose|
sequential/gru/Shape_1Shapesequential/gru/transpose:y:0*
T0*
_output_shapes
:2
sequential/gru/Shape_1
$sequential/gru/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$sequential/gru/strided_slice_1/stack
&sequential/gru/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&sequential/gru/strided_slice_1/stack_1
&sequential/gru/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&sequential/gru/strided_slice_1/stack_2È
sequential/gru/strided_slice_1StridedSlicesequential/gru/Shape_1:output:0-sequential/gru/strided_slice_1/stack:output:0/sequential/gru/strided_slice_1/stack_1:output:0/sequential/gru/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
sequential/gru/strided_slice_1£
*sequential/gru/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2,
*sequential/gru/TensorArrayV2/element_shapeî
sequential/gru/TensorArrayV2TensorListReserve3sequential/gru/TensorArrayV2/element_shape:output:0'sequential/gru/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
sequential/gru/TensorArrayV2Ý
Dsequential/gru/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2F
Dsequential/gru/TensorArrayUnstack/TensorListFromTensor/element_shape´
6sequential/gru/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorsequential/gru/transpose:y:0Msequential/gru/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type028
6sequential/gru/TensorArrayUnstack/TensorListFromTensor
$sequential/gru/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$sequential/gru/strided_slice_2/stack
&sequential/gru/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&sequential/gru/strided_slice_2/stack_1
&sequential/gru/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&sequential/gru/strided_slice_2/stack_2Ö
sequential/gru/strided_slice_2StridedSlicesequential/gru/transpose:y:0-sequential/gru/strided_slice_2/stack:output:0/sequential/gru/strided_slice_2/stack_1:output:0/sequential/gru/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2 
sequential/gru/strided_slice_2À
&sequential/gru/gru_cell/ReadVariableOpReadVariableOp/sequential_gru_gru_cell_readvariableop_resource*
_output_shapes

:`*
dtype02(
&sequential/gru/gru_cell/ReadVariableOp²
sequential/gru/gru_cell/unstackUnpack.sequential/gru/gru_cell/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num2!
sequential/gru/gru_cell/unstackÕ
-sequential/gru/gru_cell/MatMul/ReadVariableOpReadVariableOp6sequential_gru_gru_cell_matmul_readvariableop_resource*
_output_shapes

:`*
dtype02/
-sequential/gru/gru_cell/MatMul/ReadVariableOpÜ
sequential/gru/gru_cell/MatMulMatMul'sequential/gru/strided_slice_2:output:05sequential/gru/gru_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2 
sequential/gru/gru_cell/MatMulÓ
sequential/gru/gru_cell/BiasAddBiasAdd(sequential/gru/gru_cell/MatMul:product:0(sequential/gru/gru_cell/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2!
sequential/gru/gru_cell/BiasAdd
sequential/gru/gru_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
sequential/gru/gru_cell/Const
'sequential/gru/gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2)
'sequential/gru/gru_cell/split/split_dim
sequential/gru/gru_cell/splitSplit0sequential/gru/gru_cell/split/split_dim:output:0(sequential/gru/gru_cell/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
sequential/gru/gru_cell/splitÛ
/sequential/gru/gru_cell/MatMul_1/ReadVariableOpReadVariableOp8sequential_gru_gru_cell_matmul_1_readvariableop_resource*
_output_shapes

: `*
dtype021
/sequential/gru/gru_cell/MatMul_1/ReadVariableOpØ
 sequential/gru/gru_cell/MatMul_1MatMulsequential/gru/zeros:output:07sequential/gru/gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2"
 sequential/gru/gru_cell/MatMul_1Ù
!sequential/gru/gru_cell/BiasAdd_1BiasAdd*sequential/gru/gru_cell/MatMul_1:product:0(sequential/gru/gru_cell/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2#
!sequential/gru/gru_cell/BiasAdd_1
sequential/gru/gru_cell/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"        ÿÿÿÿ2!
sequential/gru/gru_cell/Const_1¡
)sequential/gru/gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2+
)sequential/gru/gru_cell/split_1/split_dimË
sequential/gru/gru_cell/split_1SplitV*sequential/gru/gru_cell/BiasAdd_1:output:0(sequential/gru/gru_cell/Const_1:output:02sequential/gru/gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2!
sequential/gru/gru_cell/split_1Ç
sequential/gru/gru_cell/addAddV2&sequential/gru/gru_cell/split:output:0(sequential/gru/gru_cell/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
sequential/gru/gru_cell/add 
sequential/gru/gru_cell/SigmoidSigmoidsequential/gru/gru_cell/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
sequential/gru/gru_cell/SigmoidË
sequential/gru/gru_cell/add_1AddV2&sequential/gru/gru_cell/split:output:1(sequential/gru/gru_cell/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
sequential/gru/gru_cell/add_1¦
!sequential/gru/gru_cell/Sigmoid_1Sigmoid!sequential/gru/gru_cell/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!sequential/gru/gru_cell/Sigmoid_1Ä
sequential/gru/gru_cell/mulMul%sequential/gru/gru_cell/Sigmoid_1:y:0(sequential/gru/gru_cell/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
sequential/gru/gru_cell/mulÂ
sequential/gru/gru_cell/add_2AddV2&sequential/gru/gru_cell/split:output:2sequential/gru/gru_cell/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
sequential/gru/gru_cell/add_2
sequential/gru/gru_cell/ReluRelu!sequential/gru/gru_cell/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
sequential/gru/gru_cell/Relu»
sequential/gru/gru_cell/mul_1Mul#sequential/gru/gru_cell/Sigmoid:y:0sequential/gru/zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
sequential/gru/gru_cell/mul_1
sequential/gru/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
sequential/gru/gru_cell/sub/xÀ
sequential/gru/gru_cell/subSub&sequential/gru/gru_cell/sub/x:output:0#sequential/gru/gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
sequential/gru/gru_cell/subÄ
sequential/gru/gru_cell/mul_2Mulsequential/gru/gru_cell/sub:z:0*sequential/gru/gru_cell/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
sequential/gru/gru_cell/mul_2¿
sequential/gru/gru_cell/add_3AddV2!sequential/gru/gru_cell/mul_1:z:0!sequential/gru/gru_cell/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
sequential/gru/gru_cell/add_3­
,sequential/gru/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2.
,sequential/gru/TensorArrayV2_1/element_shapeô
sequential/gru/TensorArrayV2_1TensorListReserve5sequential/gru/TensorArrayV2_1/element_shape:output:0'sequential/gru/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02 
sequential/gru/TensorArrayV2_1l
sequential/gru/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
sequential/gru/time
'sequential/gru/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2)
'sequential/gru/while/maximum_iterations
!sequential/gru/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2#
!sequential/gru/while/loop_counterï
sequential/gru/whileWhile*sequential/gru/while/loop_counter:output:00sequential/gru/while/maximum_iterations:output:0sequential/gru/time:output:0'sequential/gru/TensorArrayV2_1:handle:0sequential/gru/zeros:output:0'sequential/gru/strided_slice_1:output:0Fsequential/gru/TensorArrayUnstack/TensorListFromTensor:output_handle:0/sequential_gru_gru_cell_readvariableop_resource6sequential_gru_gru_cell_matmul_readvariableop_resource8sequential_gru_gru_cell_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ : : : : : *%
_read_only_resource_inputs
	**
body"R 
sequential_gru_while_body_6797**
cond"R 
sequential_gru_while_cond_6796*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ : : : : : *
parallel_iterations 2
sequential/gru/whileÓ
?sequential/gru/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2A
?sequential/gru/TensorArrayV2Stack/TensorListStack/element_shape¤
1sequential/gru/TensorArrayV2Stack/TensorListStackTensorListStacksequential/gru/while:output:3Hsequential/gru/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype023
1sequential/gru/TensorArrayV2Stack/TensorListStack
$sequential/gru/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2&
$sequential/gru/strided_slice_3/stack
&sequential/gru/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential/gru/strided_slice_3/stack_1
&sequential/gru/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&sequential/gru/strided_slice_3/stack_2ô
sequential/gru/strided_slice_3StridedSlice:sequential/gru/TensorArrayV2Stack/TensorListStack:tensor:0-sequential/gru/strided_slice_3/stack:output:0/sequential/gru/strided_slice_3/stack_1:output:0/sequential/gru/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2 
sequential/gru/strided_slice_3
sequential/gru/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2!
sequential/gru/transpose_1/permá
sequential/gru/transpose_1	Transpose:sequential/gru/TensorArrayV2Stack/TensorListStack:tensor:0(sequential/gru/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
sequential/gru/transpose_1
sequential/gru/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
sequential/gru/runtime
sequential/dropout/IdentityIdentitysequential/gru/transpose_1:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
sequential/dropout/Identity
sequential/gru_1/ShapeShape$sequential/dropout/Identity:output:0*
T0*
_output_shapes
:2
sequential/gru_1/Shape
$sequential/gru_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$sequential/gru_1/strided_slice/stack
&sequential/gru_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&sequential/gru_1/strided_slice/stack_1
&sequential/gru_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&sequential/gru_1/strided_slice/stack_2È
sequential/gru_1/strided_sliceStridedSlicesequential/gru_1/Shape:output:0-sequential/gru_1/strided_slice/stack:output:0/sequential/gru_1/strided_slice/stack_1:output:0/sequential/gru_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
sequential/gru_1/strided_slice~
sequential/gru_1/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
sequential/gru_1/zeros/mul/y°
sequential/gru_1/zeros/mulMul'sequential/gru_1/strided_slice:output:0%sequential/gru_1/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
sequential/gru_1/zeros/mul
sequential/gru_1/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
sequential/gru_1/zeros/Less/y«
sequential/gru_1/zeros/LessLesssequential/gru_1/zeros/mul:z:0&sequential/gru_1/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
sequential/gru_1/zeros/Less
sequential/gru_1/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2!
sequential/gru_1/zeros/packed/1Ç
sequential/gru_1/zeros/packedPack'sequential/gru_1/strided_slice:output:0(sequential/gru_1/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
sequential/gru_1/zeros/packed
sequential/gru_1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
sequential/gru_1/zeros/Const¹
sequential/gru_1/zerosFill&sequential/gru_1/zeros/packed:output:0%sequential/gru_1/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential/gru_1/zeros
sequential/gru_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2!
sequential/gru_1/transpose/permË
sequential/gru_1/transpose	Transpose$sequential/dropout/Identity:output:0(sequential/gru_1/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
sequential/gru_1/transpose
sequential/gru_1/Shape_1Shapesequential/gru_1/transpose:y:0*
T0*
_output_shapes
:2
sequential/gru_1/Shape_1
&sequential/gru_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential/gru_1/strided_slice_1/stack
(sequential/gru_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(sequential/gru_1/strided_slice_1/stack_1
(sequential/gru_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(sequential/gru_1/strided_slice_1/stack_2Ô
 sequential/gru_1/strided_slice_1StridedSlice!sequential/gru_1/Shape_1:output:0/sequential/gru_1/strided_slice_1/stack:output:01sequential/gru_1/strided_slice_1/stack_1:output:01sequential/gru_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 sequential/gru_1/strided_slice_1§
,sequential/gru_1/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2.
,sequential/gru_1/TensorArrayV2/element_shapeö
sequential/gru_1/TensorArrayV2TensorListReserve5sequential/gru_1/TensorArrayV2/element_shape:output:0)sequential/gru_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02 
sequential/gru_1/TensorArrayV2á
Fsequential/gru_1/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2H
Fsequential/gru_1/TensorArrayUnstack/TensorListFromTensor/element_shape¼
8sequential/gru_1/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorsequential/gru_1/transpose:y:0Osequential/gru_1/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02:
8sequential/gru_1/TensorArrayUnstack/TensorListFromTensor
&sequential/gru_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential/gru_1/strided_slice_2/stack
(sequential/gru_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(sequential/gru_1/strided_slice_2/stack_1
(sequential/gru_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(sequential/gru_1/strided_slice_2/stack_2â
 sequential/gru_1/strided_slice_2StridedSlicesequential/gru_1/transpose:y:0/sequential/gru_1/strided_slice_2/stack:output:01sequential/gru_1/strided_slice_2/stack_1:output:01sequential/gru_1/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2"
 sequential/gru_1/strided_slice_2Ì
*sequential/gru_1/gru_cell_1/ReadVariableOpReadVariableOp3sequential_gru_1_gru_cell_1_readvariableop_resource*
_output_shapes

:0*
dtype02,
*sequential/gru_1/gru_cell_1/ReadVariableOp¾
#sequential/gru_1/gru_cell_1/unstackUnpack2sequential/gru_1/gru_cell_1/ReadVariableOp:value:0*
T0* 
_output_shapes
:0:0*	
num2%
#sequential/gru_1/gru_cell_1/unstacká
1sequential/gru_1/gru_cell_1/MatMul/ReadVariableOpReadVariableOp:sequential_gru_1_gru_cell_1_matmul_readvariableop_resource*
_output_shapes

: 0*
dtype023
1sequential/gru_1/gru_cell_1/MatMul/ReadVariableOpê
"sequential/gru_1/gru_cell_1/MatMulMatMul)sequential/gru_1/strided_slice_2:output:09sequential/gru_1/gru_cell_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02$
"sequential/gru_1/gru_cell_1/MatMulã
#sequential/gru_1/gru_cell_1/BiasAddBiasAdd,sequential/gru_1/gru_cell_1/MatMul:product:0,sequential/gru_1/gru_cell_1/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02%
#sequential/gru_1/gru_cell_1/BiasAdd
!sequential/gru_1/gru_cell_1/ConstConst*
_output_shapes
: *
dtype0*
value	B :2#
!sequential/gru_1/gru_cell_1/Const¥
+sequential/gru_1/gru_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2-
+sequential/gru_1/gru_cell_1/split/split_dim
!sequential/gru_1/gru_cell_1/splitSplit4sequential/gru_1/gru_cell_1/split/split_dim:output:0,sequential/gru_1/gru_cell_1/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2#
!sequential/gru_1/gru_cell_1/splitç
3sequential/gru_1/gru_cell_1/MatMul_1/ReadVariableOpReadVariableOp<sequential_gru_1_gru_cell_1_matmul_1_readvariableop_resource*
_output_shapes

:0*
dtype025
3sequential/gru_1/gru_cell_1/MatMul_1/ReadVariableOpæ
$sequential/gru_1/gru_cell_1/MatMul_1MatMulsequential/gru_1/zeros:output:0;sequential/gru_1/gru_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02&
$sequential/gru_1/gru_cell_1/MatMul_1é
%sequential/gru_1/gru_cell_1/BiasAdd_1BiasAdd.sequential/gru_1/gru_cell_1/MatMul_1:product:0,sequential/gru_1/gru_cell_1/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02'
%sequential/gru_1/gru_cell_1/BiasAdd_1
#sequential/gru_1/gru_cell_1/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      ÿÿÿÿ2%
#sequential/gru_1/gru_cell_1/Const_1©
-sequential/gru_1/gru_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2/
-sequential/gru_1/gru_cell_1/split_1/split_dimß
#sequential/gru_1/gru_cell_1/split_1SplitV.sequential/gru_1/gru_cell_1/BiasAdd_1:output:0,sequential/gru_1/gru_cell_1/Const_1:output:06sequential/gru_1/gru_cell_1/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2%
#sequential/gru_1/gru_cell_1/split_1×
sequential/gru_1/gru_cell_1/addAddV2*sequential/gru_1/gru_cell_1/split:output:0,sequential/gru_1/gru_cell_1/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
sequential/gru_1/gru_cell_1/add¬
#sequential/gru_1/gru_cell_1/SigmoidSigmoid#sequential/gru_1/gru_cell_1/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#sequential/gru_1/gru_cell_1/SigmoidÛ
!sequential/gru_1/gru_cell_1/add_1AddV2*sequential/gru_1/gru_cell_1/split:output:1,sequential/gru_1/gru_cell_1/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!sequential/gru_1/gru_cell_1/add_1²
%sequential/gru_1/gru_cell_1/Sigmoid_1Sigmoid%sequential/gru_1/gru_cell_1/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%sequential/gru_1/gru_cell_1/Sigmoid_1Ô
sequential/gru_1/gru_cell_1/mulMul)sequential/gru_1/gru_cell_1/Sigmoid_1:y:0,sequential/gru_1/gru_cell_1/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
sequential/gru_1/gru_cell_1/mulÒ
!sequential/gru_1/gru_cell_1/add_2AddV2*sequential/gru_1/gru_cell_1/split:output:2#sequential/gru_1/gru_cell_1/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!sequential/gru_1/gru_cell_1/add_2¥
 sequential/gru_1/gru_cell_1/ReluRelu%sequential/gru_1/gru_cell_1/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 sequential/gru_1/gru_cell_1/ReluÉ
!sequential/gru_1/gru_cell_1/mul_1Mul'sequential/gru_1/gru_cell_1/Sigmoid:y:0sequential/gru_1/zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!sequential/gru_1/gru_cell_1/mul_1
!sequential/gru_1/gru_cell_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2#
!sequential/gru_1/gru_cell_1/sub/xÐ
sequential/gru_1/gru_cell_1/subSub*sequential/gru_1/gru_cell_1/sub/x:output:0'sequential/gru_1/gru_cell_1/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
sequential/gru_1/gru_cell_1/subÔ
!sequential/gru_1/gru_cell_1/mul_2Mul#sequential/gru_1/gru_cell_1/sub:z:0.sequential/gru_1/gru_cell_1/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!sequential/gru_1/gru_cell_1/mul_2Ï
!sequential/gru_1/gru_cell_1/add_3AddV2%sequential/gru_1/gru_cell_1/mul_1:z:0%sequential/gru_1/gru_cell_1/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!sequential/gru_1/gru_cell_1/add_3±
.sequential/gru_1/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   20
.sequential/gru_1/TensorArrayV2_1/element_shapeü
 sequential/gru_1/TensorArrayV2_1TensorListReserve7sequential/gru_1/TensorArrayV2_1/element_shape:output:0)sequential/gru_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02"
 sequential/gru_1/TensorArrayV2_1p
sequential/gru_1/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
sequential/gru_1/time¡
)sequential/gru_1/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2+
)sequential/gru_1/while/maximum_iterations
#sequential/gru_1/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2%
#sequential/gru_1/while/loop_counter
sequential/gru_1/whileWhile,sequential/gru_1/while/loop_counter:output:02sequential/gru_1/while/maximum_iterations:output:0sequential/gru_1/time:output:0)sequential/gru_1/TensorArrayV2_1:handle:0sequential/gru_1/zeros:output:0)sequential/gru_1/strided_slice_1:output:0Hsequential/gru_1/TensorArrayUnstack/TensorListFromTensor:output_handle:03sequential_gru_1_gru_cell_1_readvariableop_resource:sequential_gru_1_gru_cell_1_matmul_readvariableop_resource<sequential_gru_1_gru_cell_1_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	*,
body$R"
 sequential_gru_1_while_body_6953*,
cond$R"
 sequential_gru_1_while_cond_6952*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 2
sequential/gru_1/while×
Asequential/gru_1/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2C
Asequential/gru_1/TensorArrayV2Stack/TensorListStack/element_shape¬
3sequential/gru_1/TensorArrayV2Stack/TensorListStackTensorListStacksequential/gru_1/while:output:3Jsequential/gru_1/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype025
3sequential/gru_1/TensorArrayV2Stack/TensorListStack£
&sequential/gru_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2(
&sequential/gru_1/strided_slice_3/stack
(sequential/gru_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(sequential/gru_1/strided_slice_3/stack_1
(sequential/gru_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(sequential/gru_1/strided_slice_3/stack_2
 sequential/gru_1/strided_slice_3StridedSlice<sequential/gru_1/TensorArrayV2Stack/TensorListStack:tensor:0/sequential/gru_1/strided_slice_3/stack:output:01sequential/gru_1/strided_slice_3/stack_1:output:01sequential/gru_1/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2"
 sequential/gru_1/strided_slice_3
!sequential/gru_1/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2#
!sequential/gru_1/transpose_1/permé
sequential/gru_1/transpose_1	Transpose<sequential/gru_1/TensorArrayV2Stack/TensorListStack:tensor:0*sequential/gru_1/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential/gru_1/transpose_1
sequential/gru_1/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
sequential/gru_1/runtime§
sequential/dropout_1/IdentityIdentity)sequential/gru_1/strided_slice_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential/dropout_1/IdentityÀ
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource*
_output_shapes

:*
dtype02(
&sequential/dense/MatMul/ReadVariableOpÆ
sequential/dense/MatMulMatMul&sequential/dropout_1/Identity:output:0.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential/dense/MatMul¿
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'sequential/dense/BiasAdd/ReadVariableOpÅ
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential/dense/BiasAdd
IdentityIdentity!sequential/dense/BiasAdd:output:0(^sequential/dense/BiasAdd/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOp.^sequential/gru/gru_cell/MatMul/ReadVariableOp0^sequential/gru/gru_cell/MatMul_1/ReadVariableOp'^sequential/gru/gru_cell/ReadVariableOp^sequential/gru/while2^sequential/gru_1/gru_cell_1/MatMul/ReadVariableOp4^sequential/gru_1/gru_cell_1/MatMul_1/ReadVariableOp+^sequential/gru_1/gru_cell_1/ReadVariableOp^sequential/gru_1/while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:ÿÿÿÿÿÿÿÿÿ::::::::2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2P
&sequential/dense/MatMul/ReadVariableOp&sequential/dense/MatMul/ReadVariableOp2^
-sequential/gru/gru_cell/MatMul/ReadVariableOp-sequential/gru/gru_cell/MatMul/ReadVariableOp2b
/sequential/gru/gru_cell/MatMul_1/ReadVariableOp/sequential/gru/gru_cell/MatMul_1/ReadVariableOp2P
&sequential/gru/gru_cell/ReadVariableOp&sequential/gru/gru_cell/ReadVariableOp2,
sequential/gru/whilesequential/gru/while2f
1sequential/gru_1/gru_cell_1/MatMul/ReadVariableOp1sequential/gru_1/gru_cell_1/MatMul/ReadVariableOp2j
3sequential/gru_1/gru_cell_1/MatMul_1/ReadVariableOp3sequential/gru_1/gru_cell_1/MatMul_1/ReadVariableOp2X
*sequential/gru_1/gru_cell_1/ReadVariableOp*sequential/gru_1/gru_cell_1/ReadVariableOp20
sequential/gru_1/whilesequential/gru_1/while:V R
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#
_user_specified_name	gru_input

Ô
"__inference_signature_wrapper_9108
	gru_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity¢StatefulPartitionedCall£
StatefulPartitionedCallStatefulPartitionedCall	gru_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *(
f#R!
__inference__wrapped_model_70502
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:ÿÿÿÿÿÿÿÿÿ::::::::22
StatefulPartitionedCallStatefulPartitionedCall:V R
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#
_user_specified_name	gru_input
³!
Æ
while_body_8101
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_gru_cell_1_8123_0
while_gru_cell_1_8125_0
while_gru_cell_1_8127_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_gru_cell_1_8123
while_gru_cell_1_8125
while_gru_cell_1_8127¢(while/gru_cell_1/StatefulPartitionedCallÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÓ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem¦
(while/gru_cell_1/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_gru_cell_1_8123_0while_gru_cell_1_8125_0while_gru_cell_1_8127_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_gru_cell_1_layer_call_and_return_conditional_losses_77242*
(while/gru_cell_1/StatefulPartitionedCallõ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder1while/gru_cell_1/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1
while/IdentityIdentitywhile/add_1:z:0)^while/gru_cell_1/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity
while/Identity_1Identitywhile_while_maximum_iterations)^while/gru_cell_1/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1
while/Identity_2Identitywhile/add:z:0)^while/gru_cell_1/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2¸
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0)^while/gru_cell_1/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3À
while/Identity_4Identity1while/gru_cell_1/StatefulPartitionedCall:output:1)^while/gru_cell_1/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_4"0
while_gru_cell_1_8123while_gru_cell_1_8123_0"0
while_gru_cell_1_8125while_gru_cell_1_8125_0"0
while_gru_cell_1_8127while_gru_cell_1_8127_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :ÿÿÿÿÿÿÿÿÿ: : :::2T
(while/gru_cell_1/StatefulPartitionedCall(while/gru_cell_1/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
û

½
sequential_gru_while_cond_6796:
6sequential_gru_while_sequential_gru_while_loop_counter@
<sequential_gru_while_sequential_gru_while_maximum_iterations$
 sequential_gru_while_placeholder&
"sequential_gru_while_placeholder_1&
"sequential_gru_while_placeholder_2<
8sequential_gru_while_less_sequential_gru_strided_slice_1P
Lsequential_gru_while_sequential_gru_while_cond_6796___redundant_placeholder0P
Lsequential_gru_while_sequential_gru_while_cond_6796___redundant_placeholder1P
Lsequential_gru_while_sequential_gru_while_cond_6796___redundant_placeholder2P
Lsequential_gru_while_sequential_gru_while_cond_6796___redundant_placeholder3!
sequential_gru_while_identity
»
sequential/gru/while/LessLess sequential_gru_while_placeholder8sequential_gru_while_less_sequential_gru_strided_slice_1*
T0*
_output_shapes
: 2
sequential/gru/while/Less
sequential/gru/while/IdentityIdentitysequential/gru/while/Less:z:0*
T0
*
_output_shapes
: 2
sequential/gru/while/Identity"G
sequential_gru_while_identity&sequential/gru/while/Identity:output:0*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
:

a
B__inference_dropout_layer_call_and_return_conditional_losses_10500

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
dropout/Constw
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape¸
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dropout/GreaterEqual/yÂ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dropout/Cast~
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dropout/Mul_1i
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Å
ª
C__inference_gru_cell_layer_call_and_return_conditional_losses_11321

inputs
states_0
readvariableop_resource"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource
identity

identity_1¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp¢ReadVariableOpx
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:`*
dtype02
ReadVariableOpj
unstackUnpackReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num2	
unstack
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:`*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
MatMuls
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constm
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
split/split_dim¬
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
split
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

: `*
dtype02
MatMul_1/ReadVariableOp{
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2

MatMul_1y
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
	BiasAdd_1g
Const_1Const*
_output_shapes
:*
dtype0*!
valueB"        ÿÿÿÿ2	
Const_1q
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
split_1/split_dimÓ
split_1SplitVBiasAdd_1:output:0Const_1:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2	
split_1g
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2	
Sigmoidk
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
	Sigmoid_1d
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
mulb
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
add_2Q
ReluRelu	add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Relu^
mul_1MulSigmoid:y:0states_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
sub/x`
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
subd
mul_2Mulsub:z:0Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
mul_2_
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
add_3 
IdentityIdentity	add_3:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity¤

Identity_1Identity	add_3:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*E
_input_shapes4
2:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ :::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
states/0
ûD

while_body_8246
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0,
(while_gru_cell_readvariableop_resource_03
/while_gru_cell_matmul_readvariableop_resource_05
1while_gru_cell_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor*
&while_gru_cell_readvariableop_resource1
-while_gru_cell_matmul_readvariableop_resource3
/while_gru_cell_matmul_1_readvariableop_resource¢$while/gru_cell/MatMul/ReadVariableOp¢&while/gru_cell/MatMul_1/ReadVariableOp¢while/gru_cell/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÓ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem§
while/gru_cell/ReadVariableOpReadVariableOp(while_gru_cell_readvariableop_resource_0*
_output_shapes

:`*
dtype02
while/gru_cell/ReadVariableOp
while/gru_cell/unstackUnpack%while/gru_cell/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num2
while/gru_cell/unstack¼
$while/gru_cell/MatMul/ReadVariableOpReadVariableOp/while_gru_cell_matmul_readvariableop_resource_0*
_output_shapes

:`*
dtype02&
$while/gru_cell/MatMul/ReadVariableOpÊ
while/gru_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0,while/gru_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
while/gru_cell/MatMul¯
while/gru_cell/BiasAddBiasAddwhile/gru_cell/MatMul:product:0while/gru_cell/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
while/gru_cell/BiasAddn
while/gru_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/gru_cell/Const
while/gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2 
while/gru_cell/split/split_dimè
while/gru_cell/splitSplit'while/gru_cell/split/split_dim:output:0while/gru_cell/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
while/gru_cell/splitÂ
&while/gru_cell/MatMul_1/ReadVariableOpReadVariableOp1while_gru_cell_matmul_1_readvariableop_resource_0*
_output_shapes

: `*
dtype02(
&while/gru_cell/MatMul_1/ReadVariableOp³
while/gru_cell/MatMul_1MatMulwhile_placeholder_2.while/gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
while/gru_cell/MatMul_1µ
while/gru_cell/BiasAdd_1BiasAdd!while/gru_cell/MatMul_1:product:0while/gru_cell/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
while/gru_cell/BiasAdd_1
while/gru_cell/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"        ÿÿÿÿ2
while/gru_cell/Const_1
 while/gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2"
 while/gru_cell/split_1/split_dim
while/gru_cell/split_1SplitV!while/gru_cell/BiasAdd_1:output:0while/gru_cell/Const_1:output:0)while/gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
while/gru_cell/split_1£
while/gru_cell/addAddV2while/gru_cell/split:output:0while/gru_cell/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/add
while/gru_cell/SigmoidSigmoidwhile/gru_cell/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/Sigmoid§
while/gru_cell/add_1AddV2while/gru_cell/split:output:1while/gru_cell/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/add_1
while/gru_cell/Sigmoid_1Sigmoidwhile/gru_cell/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/Sigmoid_1 
while/gru_cell/mulMulwhile/gru_cell/Sigmoid_1:y:0while/gru_cell/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/mul
while/gru_cell/add_2AddV2while/gru_cell/split:output:2while/gru_cell/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/add_2~
while/gru_cell/ReluReluwhile/gru_cell/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/Relu
while/gru_cell/mul_1Mulwhile/gru_cell/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/mul_1q
while/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
while/gru_cell/sub/x
while/gru_cell/subSubwhile/gru_cell/sub/x:output:0while/gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/sub 
while/gru_cell/mul_2Mulwhile/gru_cell/sub:z:0!while/gru_cell/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/mul_2
while/gru_cell/add_3AddV2while/gru_cell/mul_1:z:0while/gru_cell/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/add_3Ü
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell/add_3:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1Î
while/IdentityIdentitywhile/add_1:z:0%^while/gru_cell/MatMul/ReadVariableOp'^while/gru_cell/MatMul_1/ReadVariableOp^while/gru_cell/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identityá
while/Identity_1Identitywhile_while_maximum_iterations%^while/gru_cell/MatMul/ReadVariableOp'^while/gru_cell/MatMul_1/ReadVariableOp^while/gru_cell/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1Ð
while/Identity_2Identitywhile/add:z:0%^while/gru_cell/MatMul/ReadVariableOp'^while/gru_cell/MatMul_1/ReadVariableOp^while/gru_cell/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2ý
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0%^while/gru_cell/MatMul/ReadVariableOp'^while/gru_cell/MatMul_1/ReadVariableOp^while/gru_cell/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3ì
while/Identity_4Identitywhile/gru_cell/add_3:z:0%^while/gru_cell/MatMul/ReadVariableOp'^while/gru_cell/MatMul_1/ReadVariableOp^while/gru_cell/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4"d
/while_gru_cell_matmul_1_readvariableop_resource1while_gru_cell_matmul_1_readvariableop_resource_0"`
-while_gru_cell_matmul_readvariableop_resource/while_gru_cell_matmul_readvariableop_resource_0"R
&while_gru_cell_readvariableop_resource(while_gru_cell_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :ÿÿÿÿÿÿÿÿÿ : : :::2L
$while/gru_cell/MatMul/ReadVariableOp$while/gru_cell/MatMul/ReadVariableOp2P
&while/gru_cell/MatMul_1/ReadVariableOp&while/gru_cell/MatMul_1/ReadVariableOp2>
while/gru_cell/ReadVariableOpwhile/gru_cell/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
: 
Æ
 
while_cond_7982
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_12
.while_while_cond_7982___redundant_placeholder02
.while_while_cond_7982___redundant_placeholder12
.while_while_cond_7982___redundant_placeholder22
.while_while_cond_7982___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
ßZ
Õ
@__inference_gru_1_layer_call_and_return_conditional_losses_11014
inputs_0&
"gru_cell_1_readvariableop_resource-
)gru_cell_1_matmul_readvariableop_resource/
+gru_cell_1_matmul_1_readvariableop_resource
identity¢ gru_cell_1/MatMul/ReadVariableOp¢"gru_cell_1/MatMul_1/ReadVariableOp¢gru_cell_1/ReadVariableOp¢whileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ü
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2
strided_slice_2
gru_cell_1/ReadVariableOpReadVariableOp"gru_cell_1_readvariableop_resource*
_output_shapes

:0*
dtype02
gru_cell_1/ReadVariableOp
gru_cell_1/unstackUnpack!gru_cell_1/ReadVariableOp:value:0*
T0* 
_output_shapes
:0:0*	
num2
gru_cell_1/unstack®
 gru_cell_1/MatMul/ReadVariableOpReadVariableOp)gru_cell_1_matmul_readvariableop_resource*
_output_shapes

: 0*
dtype02"
 gru_cell_1/MatMul/ReadVariableOp¦
gru_cell_1/MatMulMatMulstrided_slice_2:output:0(gru_cell_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02
gru_cell_1/MatMul
gru_cell_1/BiasAddBiasAddgru_cell_1/MatMul:product:0gru_cell_1/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02
gru_cell_1/BiasAddf
gru_cell_1/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru_cell_1/Const
gru_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
gru_cell_1/split/split_dimØ
gru_cell_1/splitSplit#gru_cell_1/split/split_dim:output:0gru_cell_1/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
gru_cell_1/split´
"gru_cell_1/MatMul_1/ReadVariableOpReadVariableOp+gru_cell_1_matmul_1_readvariableop_resource*
_output_shapes

:0*
dtype02$
"gru_cell_1/MatMul_1/ReadVariableOp¢
gru_cell_1/MatMul_1MatMulzeros:output:0*gru_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02
gru_cell_1/MatMul_1¥
gru_cell_1/BiasAdd_1BiasAddgru_cell_1/MatMul_1:product:0gru_cell_1/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02
gru_cell_1/BiasAdd_1}
gru_cell_1/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      ÿÿÿÿ2
gru_cell_1/Const_1
gru_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
gru_cell_1/split_1/split_dim
gru_cell_1/split_1SplitVgru_cell_1/BiasAdd_1:output:0gru_cell_1/Const_1:output:0%gru_cell_1/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
gru_cell_1/split_1
gru_cell_1/addAddV2gru_cell_1/split:output:0gru_cell_1/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_cell_1/addy
gru_cell_1/SigmoidSigmoidgru_cell_1/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_cell_1/Sigmoid
gru_cell_1/add_1AddV2gru_cell_1/split:output:1gru_cell_1/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_cell_1/add_1
gru_cell_1/Sigmoid_1Sigmoidgru_cell_1/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_cell_1/Sigmoid_1
gru_cell_1/mulMulgru_cell_1/Sigmoid_1:y:0gru_cell_1/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_cell_1/mul
gru_cell_1/add_2AddV2gru_cell_1/split:output:2gru_cell_1/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_cell_1/add_2r
gru_cell_1/ReluRelugru_cell_1/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_cell_1/Relu
gru_cell_1/mul_1Mulgru_cell_1/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_cell_1/mul_1i
gru_cell_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru_cell_1/sub/x
gru_cell_1/subSubgru_cell_1/sub/x:output:0gru_cell_1/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_cell_1/sub
gru_cell_1/mul_2Mulgru_cell_1/sub:z:0gru_cell_1/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_cell_1/mul_2
gru_cell_1/add_3AddV2gru_cell_1/mul_1:z:0gru_cell_1/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_cell_1/add_3
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
TensorArrayV2_1/element_shape¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter¥
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0"gru_cell_1_readvariableop_resource)gru_cell_1_matmul_readvariableop_resource+gru_cell_1_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_10924*
condR
while_cond_10923*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   22
0TensorArrayV2Stack/TensorListStack/element_shapeñ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm®
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeØ
IdentityIdentitystrided_slice_3:output:0!^gru_cell_1/MatMul/ReadVariableOp#^gru_cell_1/MatMul_1/ReadVariableOp^gru_cell_1/ReadVariableOp^while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ :::2D
 gru_cell_1/MatMul/ReadVariableOp gru_cell_1/MatMul/ReadVariableOp2H
"gru_cell_1/MatMul_1/ReadVariableOp"gru_cell_1/MatMul_1/ReadVariableOp26
gru_cell_1/ReadVariableOpgru_cell_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
inputs/0
ïE

__inference__traced_save_11573
file_prefix+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop2
.savev2_gru_gru_cell_kernel_read_readvariableop<
8savev2_gru_gru_cell_recurrent_kernel_read_readvariableop0
,savev2_gru_gru_cell_bias_read_readvariableop6
2savev2_gru_1_gru_cell_1_kernel_read_readvariableop@
<savev2_gru_1_gru_cell_1_recurrent_kernel_read_readvariableop4
0savev2_gru_1_gru_cell_1_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop2
.savev2_adam_dense_kernel_m_read_readvariableop0
,savev2_adam_dense_bias_m_read_readvariableop9
5savev2_adam_gru_gru_cell_kernel_m_read_readvariableopC
?savev2_adam_gru_gru_cell_recurrent_kernel_m_read_readvariableop7
3savev2_adam_gru_gru_cell_bias_m_read_readvariableop=
9savev2_adam_gru_1_gru_cell_1_kernel_m_read_readvariableopG
Csavev2_adam_gru_1_gru_cell_1_recurrent_kernel_m_read_readvariableop;
7savev2_adam_gru_1_gru_cell_1_bias_m_read_readvariableop2
.savev2_adam_dense_kernel_v_read_readvariableop0
,savev2_adam_dense_bias_v_read_readvariableop9
5savev2_adam_gru_gru_cell_kernel_v_read_readvariableopC
?savev2_adam_gru_gru_cell_recurrent_kernel_v_read_readvariableop7
3savev2_adam_gru_gru_cell_bias_v_read_readvariableop=
9savev2_adam_gru_1_gru_cell_1_kernel_v_read_readvariableopG
Csavev2_adam_gru_1_gru_cell_1_recurrent_kernel_v_read_readvariableop;
7savev2_adam_gru_1_gru_cell_1_bias_v_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpoints
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameÌ
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
: *
dtype0*Þ
valueÔBÑ B6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesÈ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
: *
dtype0*S
valueJBH B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesý
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop.savev2_gru_gru_cell_kernel_read_readvariableop8savev2_gru_gru_cell_recurrent_kernel_read_readvariableop,savev2_gru_gru_cell_bias_read_readvariableop2savev2_gru_1_gru_cell_1_kernel_read_readvariableop<savev2_gru_1_gru_cell_1_recurrent_kernel_read_readvariableop0savev2_gru_1_gru_cell_1_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop.savev2_adam_dense_kernel_m_read_readvariableop,savev2_adam_dense_bias_m_read_readvariableop5savev2_adam_gru_gru_cell_kernel_m_read_readvariableop?savev2_adam_gru_gru_cell_recurrent_kernel_m_read_readvariableop3savev2_adam_gru_gru_cell_bias_m_read_readvariableop9savev2_adam_gru_1_gru_cell_1_kernel_m_read_readvariableopCsavev2_adam_gru_1_gru_cell_1_recurrent_kernel_m_read_readvariableop7savev2_adam_gru_1_gru_cell_1_bias_m_read_readvariableop.savev2_adam_dense_kernel_v_read_readvariableop,savev2_adam_dense_bias_v_read_readvariableop5savev2_adam_gru_gru_cell_kernel_v_read_readvariableop?savev2_adam_gru_gru_cell_recurrent_kernel_v_read_readvariableop3savev2_adam_gru_gru_cell_bias_v_read_readvariableop9savev2_adam_gru_1_gru_cell_1_kernel_v_read_readvariableopCsavev2_adam_gru_1_gru_cell_1_recurrent_kernel_v_read_readvariableop7savev2_adam_gru_1_gru_cell_1_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *.
dtypes$
"2 	2
SaveV2º
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes¡
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*
_input_shapesù
ö: ::: : : : : :`: `:`: 0:0:0: : :::`: `:`: 0:0:0:::`: `:`: 0:0:0: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:`:$	 

_output_shapes

: `:$
 

_output_shapes

:`:$ 

_output_shapes

: 0:$ 

_output_shapes

:0:$ 

_output_shapes

:0:

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:`:$ 

_output_shapes

: `:$ 

_output_shapes

:`:$ 

_output_shapes

: 0:$ 

_output_shapes

:0:$ 

_output_shapes

:0:$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:`:$ 

_output_shapes

: `:$ 

_output_shapes

:`:$ 

_output_shapes

: 0:$ 

_output_shapes

:0:$ 

_output_shapes

:0: 

_output_shapes
: 


%__inference_gru_1_layer_call_fn_11195
inputs_0
unknown
	unknown_0
	unknown_1
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_gru_1_layer_call_and_return_conditional_losses_81652
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ :::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
inputs/0
Æ
 
while_cond_7538
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_12
.while_while_cond_7538___redundant_placeholder02
.while_while_cond_7538___redundant_placeholder12
.while_while_cond_7538___redundant_placeholder22
.while_while_cond_7538___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
:
ûD

while_body_9877
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0,
(while_gru_cell_readvariableop_resource_03
/while_gru_cell_matmul_readvariableop_resource_05
1while_gru_cell_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor*
&while_gru_cell_readvariableop_resource1
-while_gru_cell_matmul_readvariableop_resource3
/while_gru_cell_matmul_1_readvariableop_resource¢$while/gru_cell/MatMul/ReadVariableOp¢&while/gru_cell/MatMul_1/ReadVariableOp¢while/gru_cell/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÓ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem§
while/gru_cell/ReadVariableOpReadVariableOp(while_gru_cell_readvariableop_resource_0*
_output_shapes

:`*
dtype02
while/gru_cell/ReadVariableOp
while/gru_cell/unstackUnpack%while/gru_cell/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num2
while/gru_cell/unstack¼
$while/gru_cell/MatMul/ReadVariableOpReadVariableOp/while_gru_cell_matmul_readvariableop_resource_0*
_output_shapes

:`*
dtype02&
$while/gru_cell/MatMul/ReadVariableOpÊ
while/gru_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0,while/gru_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
while/gru_cell/MatMul¯
while/gru_cell/BiasAddBiasAddwhile/gru_cell/MatMul:product:0while/gru_cell/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
while/gru_cell/BiasAddn
while/gru_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/gru_cell/Const
while/gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2 
while/gru_cell/split/split_dimè
while/gru_cell/splitSplit'while/gru_cell/split/split_dim:output:0while/gru_cell/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
while/gru_cell/splitÂ
&while/gru_cell/MatMul_1/ReadVariableOpReadVariableOp1while_gru_cell_matmul_1_readvariableop_resource_0*
_output_shapes

: `*
dtype02(
&while/gru_cell/MatMul_1/ReadVariableOp³
while/gru_cell/MatMul_1MatMulwhile_placeholder_2.while/gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
while/gru_cell/MatMul_1µ
while/gru_cell/BiasAdd_1BiasAdd!while/gru_cell/MatMul_1:product:0while/gru_cell/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
while/gru_cell/BiasAdd_1
while/gru_cell/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"        ÿÿÿÿ2
while/gru_cell/Const_1
 while/gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2"
 while/gru_cell/split_1/split_dim
while/gru_cell/split_1SplitV!while/gru_cell/BiasAdd_1:output:0while/gru_cell/Const_1:output:0)while/gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
while/gru_cell/split_1£
while/gru_cell/addAddV2while/gru_cell/split:output:0while/gru_cell/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/add
while/gru_cell/SigmoidSigmoidwhile/gru_cell/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/Sigmoid§
while/gru_cell/add_1AddV2while/gru_cell/split:output:1while/gru_cell/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/add_1
while/gru_cell/Sigmoid_1Sigmoidwhile/gru_cell/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/Sigmoid_1 
while/gru_cell/mulMulwhile/gru_cell/Sigmoid_1:y:0while/gru_cell/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/mul
while/gru_cell/add_2AddV2while/gru_cell/split:output:2while/gru_cell/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/add_2~
while/gru_cell/ReluReluwhile/gru_cell/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/Relu
while/gru_cell/mul_1Mulwhile/gru_cell/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/mul_1q
while/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
while/gru_cell/sub/x
while/gru_cell/subSubwhile/gru_cell/sub/x:output:0while/gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/sub 
while/gru_cell/mul_2Mulwhile/gru_cell/sub:z:0!while/gru_cell/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/mul_2
while/gru_cell/add_3AddV2while/gru_cell/mul_1:z:0while/gru_cell/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/gru_cell/add_3Ü
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell/add_3:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1Î
while/IdentityIdentitywhile/add_1:z:0%^while/gru_cell/MatMul/ReadVariableOp'^while/gru_cell/MatMul_1/ReadVariableOp^while/gru_cell/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identityá
while/Identity_1Identitywhile_while_maximum_iterations%^while/gru_cell/MatMul/ReadVariableOp'^while/gru_cell/MatMul_1/ReadVariableOp^while/gru_cell/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1Ð
while/Identity_2Identitywhile/add:z:0%^while/gru_cell/MatMul/ReadVariableOp'^while/gru_cell/MatMul_1/ReadVariableOp^while/gru_cell/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2ý
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0%^while/gru_cell/MatMul/ReadVariableOp'^while/gru_cell/MatMul_1/ReadVariableOp^while/gru_cell/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3ì
while/Identity_4Identitywhile/gru_cell/add_3:z:0%^while/gru_cell/MatMul/ReadVariableOp'^while/gru_cell/MatMul_1/ReadVariableOp^while/gru_cell/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/Identity_4"d
/while_gru_cell_matmul_1_readvariableop_resource1while_gru_cell_matmul_1_readvariableop_resource_0"`
-while_gru_cell_matmul_readvariableop_resource/while_gru_cell_matmul_readvariableop_resource_0"R
&while_gru_cell_readvariableop_resource(while_gru_cell_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :ÿÿÿÿÿÿÿÿÿ : : :::2L
$while/gru_cell/MatMul/ReadVariableOp$while/gru_cell/MatMul/ReadVariableOp2P
&while/gru_cell/MatMul_1/ReadVariableOp&while/gru_cell/MatMul_1/ReadVariableOp2>
while/gru_cell/ReadVariableOpwhile/gru_cell/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
: 
òF
£
while_body_8623
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0.
*while_gru_cell_1_readvariableop_resource_05
1while_gru_cell_1_matmul_readvariableop_resource_07
3while_gru_cell_1_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor,
(while_gru_cell_1_readvariableop_resource3
/while_gru_cell_1_matmul_readvariableop_resource5
1while_gru_cell_1_matmul_1_readvariableop_resource¢&while/gru_cell_1/MatMul/ReadVariableOp¢(while/gru_cell_1/MatMul_1/ReadVariableOp¢while/gru_cell_1/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÓ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem­
while/gru_cell_1/ReadVariableOpReadVariableOp*while_gru_cell_1_readvariableop_resource_0*
_output_shapes

:0*
dtype02!
while/gru_cell_1/ReadVariableOp
while/gru_cell_1/unstackUnpack'while/gru_cell_1/ReadVariableOp:value:0*
T0* 
_output_shapes
:0:0*	
num2
while/gru_cell_1/unstackÂ
&while/gru_cell_1/MatMul/ReadVariableOpReadVariableOp1while_gru_cell_1_matmul_readvariableop_resource_0*
_output_shapes

: 0*
dtype02(
&while/gru_cell_1/MatMul/ReadVariableOpÐ
while/gru_cell_1/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/gru_cell_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02
while/gru_cell_1/MatMul·
while/gru_cell_1/BiasAddBiasAdd!while/gru_cell_1/MatMul:product:0!while/gru_cell_1/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02
while/gru_cell_1/BiasAddr
while/gru_cell_1/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/gru_cell_1/Const
 while/gru_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2"
 while/gru_cell_1/split/split_dimð
while/gru_cell_1/splitSplit)while/gru_cell_1/split/split_dim:output:0!while/gru_cell_1/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
while/gru_cell_1/splitÈ
(while/gru_cell_1/MatMul_1/ReadVariableOpReadVariableOp3while_gru_cell_1_matmul_1_readvariableop_resource_0*
_output_shapes

:0*
dtype02*
(while/gru_cell_1/MatMul_1/ReadVariableOp¹
while/gru_cell_1/MatMul_1MatMulwhile_placeholder_20while/gru_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02
while/gru_cell_1/MatMul_1½
while/gru_cell_1/BiasAdd_1BiasAdd#while/gru_cell_1/MatMul_1:product:0!while/gru_cell_1/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02
while/gru_cell_1/BiasAdd_1
while/gru_cell_1/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      ÿÿÿÿ2
while/gru_cell_1/Const_1
"while/gru_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2$
"while/gru_cell_1/split_1/split_dim¨
while/gru_cell_1/split_1SplitV#while/gru_cell_1/BiasAdd_1:output:0!while/gru_cell_1/Const_1:output:0+while/gru_cell_1/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
while/gru_cell_1/split_1«
while/gru_cell_1/addAddV2while/gru_cell_1/split:output:0!while/gru_cell_1/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_1/add
while/gru_cell_1/SigmoidSigmoidwhile/gru_cell_1/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_1/Sigmoid¯
while/gru_cell_1/add_1AddV2while/gru_cell_1/split:output:1!while/gru_cell_1/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_1/add_1
while/gru_cell_1/Sigmoid_1Sigmoidwhile/gru_cell_1/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_1/Sigmoid_1¨
while/gru_cell_1/mulMulwhile/gru_cell_1/Sigmoid_1:y:0!while/gru_cell_1/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_1/mul¦
while/gru_cell_1/add_2AddV2while/gru_cell_1/split:output:2while/gru_cell_1/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_1/add_2
while/gru_cell_1/ReluReluwhile/gru_cell_1/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_1/Relu
while/gru_cell_1/mul_1Mulwhile/gru_cell_1/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_1/mul_1u
while/gru_cell_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
while/gru_cell_1/sub/x¤
while/gru_cell_1/subSubwhile/gru_cell_1/sub/x:output:0while/gru_cell_1/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_1/sub¨
while/gru_cell_1/mul_2Mulwhile/gru_cell_1/sub:z:0#while/gru_cell_1/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_1/mul_2£
while/gru_cell_1/add_3AddV2while/gru_cell_1/mul_1:z:0while/gru_cell_1/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_1/add_3Þ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_1/add_3:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1Ô
while/IdentityIdentitywhile/add_1:z:0'^while/gru_cell_1/MatMul/ReadVariableOp)^while/gru_cell_1/MatMul_1/ReadVariableOp ^while/gru_cell_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identityç
while/Identity_1Identitywhile_while_maximum_iterations'^while/gru_cell_1/MatMul/ReadVariableOp)^while/gru_cell_1/MatMul_1/ReadVariableOp ^while/gru_cell_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1Ö
while/Identity_2Identitywhile/add:z:0'^while/gru_cell_1/MatMul/ReadVariableOp)^while/gru_cell_1/MatMul_1/ReadVariableOp ^while/gru_cell_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0'^while/gru_cell_1/MatMul/ReadVariableOp)^while/gru_cell_1/MatMul_1/ReadVariableOp ^while/gru_cell_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3ô
while/Identity_4Identitywhile/gru_cell_1/add_3:z:0'^while/gru_cell_1/MatMul/ReadVariableOp)^while/gru_cell_1/MatMul_1/ReadVariableOp ^while/gru_cell_1/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_4"h
1while_gru_cell_1_matmul_1_readvariableop_resource3while_gru_cell_1_matmul_1_readvariableop_resource_0"d
/while_gru_cell_1_matmul_readvariableop_resource1while_gru_cell_1_matmul_readvariableop_resource_0"V
(while_gru_cell_1_readvariableop_resource*while_gru_cell_1_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :ÿÿÿÿÿÿÿÿÿ: : :::2P
&while/gru_cell_1/MatMul/ReadVariableOp&while/gru_cell_1/MatMul/ReadVariableOp2T
(while/gru_cell_1/MatMul_1/ReadVariableOp(while/gru_cell_1/MatMul_1/ReadVariableOp2B
while/gru_cell_1/ReadVariableOpwhile/gru_cell_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 

c
D__inference_dropout_1_layer_call_and_return_conditional_losses_11207

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape´
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dropout/GreaterEqual/y¾
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
»
ì
gru_while_cond_9512$
 gru_while_gru_while_loop_counter*
&gru_while_gru_while_maximum_iterations
gru_while_placeholder
gru_while_placeholder_1
gru_while_placeholder_2&
"gru_while_less_gru_strided_slice_1:
6gru_while_gru_while_cond_9512___redundant_placeholder0:
6gru_while_gru_while_cond_9512___redundant_placeholder1:
6gru_while_gru_while_cond_9512___redundant_placeholder2:
6gru_while_gru_while_cond_9512___redundant_placeholder3
gru_while_identity

gru/while/LessLessgru_while_placeholder"gru_while_less_gru_strided_slice_1*
T0*
_output_shapes
: 2
gru/while/Lessi
gru/while/IdentityIdentitygru/while/Less:z:0*
T0
*
_output_shapes
: 2
gru/while/Identity"1
gru_while_identitygru/while/Identity:output:0*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
:
Ë
¥
while_cond_10583
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_10583___redundant_placeholder03
/while_while_cond_10583___redundant_placeholder13
/while_while_cond_10583___redundant_placeholder23
/while_while_cond_10583___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
b
¡
 sequential_gru_1_while_body_6953>
:sequential_gru_1_while_sequential_gru_1_while_loop_counterD
@sequential_gru_1_while_sequential_gru_1_while_maximum_iterations&
"sequential_gru_1_while_placeholder(
$sequential_gru_1_while_placeholder_1(
$sequential_gru_1_while_placeholder_2=
9sequential_gru_1_while_sequential_gru_1_strided_slice_1_0y
usequential_gru_1_while_tensorarrayv2read_tensorlistgetitem_sequential_gru_1_tensorarrayunstack_tensorlistfromtensor_0?
;sequential_gru_1_while_gru_cell_1_readvariableop_resource_0F
Bsequential_gru_1_while_gru_cell_1_matmul_readvariableop_resource_0H
Dsequential_gru_1_while_gru_cell_1_matmul_1_readvariableop_resource_0#
sequential_gru_1_while_identity%
!sequential_gru_1_while_identity_1%
!sequential_gru_1_while_identity_2%
!sequential_gru_1_while_identity_3%
!sequential_gru_1_while_identity_4;
7sequential_gru_1_while_sequential_gru_1_strided_slice_1w
ssequential_gru_1_while_tensorarrayv2read_tensorlistgetitem_sequential_gru_1_tensorarrayunstack_tensorlistfromtensor=
9sequential_gru_1_while_gru_cell_1_readvariableop_resourceD
@sequential_gru_1_while_gru_cell_1_matmul_readvariableop_resourceF
Bsequential_gru_1_while_gru_cell_1_matmul_1_readvariableop_resource¢7sequential/gru_1/while/gru_cell_1/MatMul/ReadVariableOp¢9sequential/gru_1/while/gru_cell_1/MatMul_1/ReadVariableOp¢0sequential/gru_1/while/gru_cell_1/ReadVariableOpå
Hsequential/gru_1/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2J
Hsequential/gru_1/while/TensorArrayV2Read/TensorListGetItem/element_shape¹
:sequential/gru_1/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemusequential_gru_1_while_tensorarrayv2read_tensorlistgetitem_sequential_gru_1_tensorarrayunstack_tensorlistfromtensor_0"sequential_gru_1_while_placeholderQsequential/gru_1/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype02<
:sequential/gru_1/while/TensorArrayV2Read/TensorListGetItemà
0sequential/gru_1/while/gru_cell_1/ReadVariableOpReadVariableOp;sequential_gru_1_while_gru_cell_1_readvariableop_resource_0*
_output_shapes

:0*
dtype022
0sequential/gru_1/while/gru_cell_1/ReadVariableOpÐ
)sequential/gru_1/while/gru_cell_1/unstackUnpack8sequential/gru_1/while/gru_cell_1/ReadVariableOp:value:0*
T0* 
_output_shapes
:0:0*	
num2+
)sequential/gru_1/while/gru_cell_1/unstackõ
7sequential/gru_1/while/gru_cell_1/MatMul/ReadVariableOpReadVariableOpBsequential_gru_1_while_gru_cell_1_matmul_readvariableop_resource_0*
_output_shapes

: 0*
dtype029
7sequential/gru_1/while/gru_cell_1/MatMul/ReadVariableOp
(sequential/gru_1/while/gru_cell_1/MatMulMatMulAsequential/gru_1/while/TensorArrayV2Read/TensorListGetItem:item:0?sequential/gru_1/while/gru_cell_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02*
(sequential/gru_1/while/gru_cell_1/MatMulû
)sequential/gru_1/while/gru_cell_1/BiasAddBiasAdd2sequential/gru_1/while/gru_cell_1/MatMul:product:02sequential/gru_1/while/gru_cell_1/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02+
)sequential/gru_1/while/gru_cell_1/BiasAdd
'sequential/gru_1/while/gru_cell_1/ConstConst*
_output_shapes
: *
dtype0*
value	B :2)
'sequential/gru_1/while/gru_cell_1/Const±
1sequential/gru_1/while/gru_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ23
1sequential/gru_1/while/gru_cell_1/split/split_dim´
'sequential/gru_1/while/gru_cell_1/splitSplit:sequential/gru_1/while/gru_cell_1/split/split_dim:output:02sequential/gru_1/while/gru_cell_1/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2)
'sequential/gru_1/while/gru_cell_1/splitû
9sequential/gru_1/while/gru_cell_1/MatMul_1/ReadVariableOpReadVariableOpDsequential_gru_1_while_gru_cell_1_matmul_1_readvariableop_resource_0*
_output_shapes

:0*
dtype02;
9sequential/gru_1/while/gru_cell_1/MatMul_1/ReadVariableOpý
*sequential/gru_1/while/gru_cell_1/MatMul_1MatMul$sequential_gru_1_while_placeholder_2Asequential/gru_1/while/gru_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02,
*sequential/gru_1/while/gru_cell_1/MatMul_1
+sequential/gru_1/while/gru_cell_1/BiasAdd_1BiasAdd4sequential/gru_1/while/gru_cell_1/MatMul_1:product:02sequential/gru_1/while/gru_cell_1/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02-
+sequential/gru_1/while/gru_cell_1/BiasAdd_1«
)sequential/gru_1/while/gru_cell_1/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      ÿÿÿÿ2+
)sequential/gru_1/while/gru_cell_1/Const_1µ
3sequential/gru_1/while/gru_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ25
3sequential/gru_1/while/gru_cell_1/split_1/split_dimý
)sequential/gru_1/while/gru_cell_1/split_1SplitV4sequential/gru_1/while/gru_cell_1/BiasAdd_1:output:02sequential/gru_1/while/gru_cell_1/Const_1:output:0<sequential/gru_1/while/gru_cell_1/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2+
)sequential/gru_1/while/gru_cell_1/split_1ï
%sequential/gru_1/while/gru_cell_1/addAddV20sequential/gru_1/while/gru_cell_1/split:output:02sequential/gru_1/while/gru_cell_1/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%sequential/gru_1/while/gru_cell_1/add¾
)sequential/gru_1/while/gru_cell_1/SigmoidSigmoid)sequential/gru_1/while/gru_cell_1/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)sequential/gru_1/while/gru_cell_1/Sigmoidó
'sequential/gru_1/while/gru_cell_1/add_1AddV20sequential/gru_1/while/gru_cell_1/split:output:12sequential/gru_1/while/gru_cell_1/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'sequential/gru_1/while/gru_cell_1/add_1Ä
+sequential/gru_1/while/gru_cell_1/Sigmoid_1Sigmoid+sequential/gru_1/while/gru_cell_1/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2-
+sequential/gru_1/while/gru_cell_1/Sigmoid_1ì
%sequential/gru_1/while/gru_cell_1/mulMul/sequential/gru_1/while/gru_cell_1/Sigmoid_1:y:02sequential/gru_1/while/gru_cell_1/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%sequential/gru_1/while/gru_cell_1/mulê
'sequential/gru_1/while/gru_cell_1/add_2AddV20sequential/gru_1/while/gru_cell_1/split:output:2)sequential/gru_1/while/gru_cell_1/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'sequential/gru_1/while/gru_cell_1/add_2·
&sequential/gru_1/while/gru_cell_1/ReluRelu+sequential/gru_1/while/gru_cell_1/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&sequential/gru_1/while/gru_cell_1/Reluà
'sequential/gru_1/while/gru_cell_1/mul_1Mul-sequential/gru_1/while/gru_cell_1/Sigmoid:y:0$sequential_gru_1_while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'sequential/gru_1/while/gru_cell_1/mul_1
'sequential/gru_1/while/gru_cell_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2)
'sequential/gru_1/while/gru_cell_1/sub/xè
%sequential/gru_1/while/gru_cell_1/subSub0sequential/gru_1/while/gru_cell_1/sub/x:output:0-sequential/gru_1/while/gru_cell_1/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%sequential/gru_1/while/gru_cell_1/subì
'sequential/gru_1/while/gru_cell_1/mul_2Mul)sequential/gru_1/while/gru_cell_1/sub:z:04sequential/gru_1/while/gru_cell_1/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'sequential/gru_1/while/gru_cell_1/mul_2ç
'sequential/gru_1/while/gru_cell_1/add_3AddV2+sequential/gru_1/while/gru_cell_1/mul_1:z:0+sequential/gru_1/while/gru_cell_1/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'sequential/gru_1/while/gru_cell_1/add_3³
;sequential/gru_1/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem$sequential_gru_1_while_placeholder_1"sequential_gru_1_while_placeholder+sequential/gru_1/while/gru_cell_1/add_3:z:0*
_output_shapes
: *
element_dtype02=
;sequential/gru_1/while/TensorArrayV2Write/TensorListSetItem~
sequential/gru_1/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
sequential/gru_1/while/add/y­
sequential/gru_1/while/addAddV2"sequential_gru_1_while_placeholder%sequential/gru_1/while/add/y:output:0*
T0*
_output_shapes
: 2
sequential/gru_1/while/add
sequential/gru_1/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2 
sequential/gru_1/while/add_1/yË
sequential/gru_1/while/add_1AddV2:sequential_gru_1_while_sequential_gru_1_while_loop_counter'sequential/gru_1/while/add_1/y:output:0*
T0*
_output_shapes
: 2
sequential/gru_1/while/add_1º
sequential/gru_1/while/IdentityIdentity sequential/gru_1/while/add_1:z:08^sequential/gru_1/while/gru_cell_1/MatMul/ReadVariableOp:^sequential/gru_1/while/gru_cell_1/MatMul_1/ReadVariableOp1^sequential/gru_1/while/gru_cell_1/ReadVariableOp*
T0*
_output_shapes
: 2!
sequential/gru_1/while/IdentityÞ
!sequential/gru_1/while/Identity_1Identity@sequential_gru_1_while_sequential_gru_1_while_maximum_iterations8^sequential/gru_1/while/gru_cell_1/MatMul/ReadVariableOp:^sequential/gru_1/while/gru_cell_1/MatMul_1/ReadVariableOp1^sequential/gru_1/while/gru_cell_1/ReadVariableOp*
T0*
_output_shapes
: 2#
!sequential/gru_1/while/Identity_1¼
!sequential/gru_1/while/Identity_2Identitysequential/gru_1/while/add:z:08^sequential/gru_1/while/gru_cell_1/MatMul/ReadVariableOp:^sequential/gru_1/while/gru_cell_1/MatMul_1/ReadVariableOp1^sequential/gru_1/while/gru_cell_1/ReadVariableOp*
T0*
_output_shapes
: 2#
!sequential/gru_1/while/Identity_2é
!sequential/gru_1/while/Identity_3IdentityKsequential/gru_1/while/TensorArrayV2Write/TensorListSetItem:output_handle:08^sequential/gru_1/while/gru_cell_1/MatMul/ReadVariableOp:^sequential/gru_1/while/gru_cell_1/MatMul_1/ReadVariableOp1^sequential/gru_1/while/gru_cell_1/ReadVariableOp*
T0*
_output_shapes
: 2#
!sequential/gru_1/while/Identity_3Ú
!sequential/gru_1/while/Identity_4Identity+sequential/gru_1/while/gru_cell_1/add_3:z:08^sequential/gru_1/while/gru_cell_1/MatMul/ReadVariableOp:^sequential/gru_1/while/gru_cell_1/MatMul_1/ReadVariableOp1^sequential/gru_1/while/gru_cell_1/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!sequential/gru_1/while/Identity_4"
Bsequential_gru_1_while_gru_cell_1_matmul_1_readvariableop_resourceDsequential_gru_1_while_gru_cell_1_matmul_1_readvariableop_resource_0"
@sequential_gru_1_while_gru_cell_1_matmul_readvariableop_resourceBsequential_gru_1_while_gru_cell_1_matmul_readvariableop_resource_0"x
9sequential_gru_1_while_gru_cell_1_readvariableop_resource;sequential_gru_1_while_gru_cell_1_readvariableop_resource_0"K
sequential_gru_1_while_identity(sequential/gru_1/while/Identity:output:0"O
!sequential_gru_1_while_identity_1*sequential/gru_1/while/Identity_1:output:0"O
!sequential_gru_1_while_identity_2*sequential/gru_1/while/Identity_2:output:0"O
!sequential_gru_1_while_identity_3*sequential/gru_1/while/Identity_3:output:0"O
!sequential_gru_1_while_identity_4*sequential/gru_1/while/Identity_4:output:0"t
7sequential_gru_1_while_sequential_gru_1_strided_slice_19sequential_gru_1_while_sequential_gru_1_strided_slice_1_0"ì
ssequential_gru_1_while_tensorarrayv2read_tensorlistgetitem_sequential_gru_1_tensorarrayunstack_tensorlistfromtensorusequential_gru_1_while_tensorarrayv2read_tensorlistgetitem_sequential_gru_1_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :ÿÿÿÿÿÿÿÿÿ: : :::2r
7sequential/gru_1/while/gru_cell_1/MatMul/ReadVariableOp7sequential/gru_1/while/gru_cell_1/MatMul/ReadVariableOp2v
9sequential/gru_1/while/gru_cell_1/MatMul_1/ReadVariableOp9sequential/gru_1/while/gru_cell_1/MatMul_1/ReadVariableOp2d
0sequential/gru_1/while/gru_cell_1/ReadVariableOp0sequential/gru_1/while/gru_cell_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
±K

gru_while_body_9177$
 gru_while_gru_while_loop_counter*
&gru_while_gru_while_maximum_iterations
gru_while_placeholder
gru_while_placeholder_1
gru_while_placeholder_2#
gru_while_gru_strided_slice_1_0_
[gru_while_tensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensor_00
,gru_while_gru_cell_readvariableop_resource_07
3gru_while_gru_cell_matmul_readvariableop_resource_09
5gru_while_gru_cell_matmul_1_readvariableop_resource_0
gru_while_identity
gru_while_identity_1
gru_while_identity_2
gru_while_identity_3
gru_while_identity_4!
gru_while_gru_strided_slice_1]
Ygru_while_tensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensor.
*gru_while_gru_cell_readvariableop_resource5
1gru_while_gru_cell_matmul_readvariableop_resource7
3gru_while_gru_cell_matmul_1_readvariableop_resource¢(gru/while/gru_cell/MatMul/ReadVariableOp¢*gru/while/gru_cell/MatMul_1/ReadVariableOp¢!gru/while/gru_cell/ReadVariableOpË
;gru/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2=
;gru/while/TensorArrayV2Read/TensorListGetItem/element_shapeë
-gru/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem[gru_while_tensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensor_0gru_while_placeholderDgru/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02/
-gru/while/TensorArrayV2Read/TensorListGetItem³
!gru/while/gru_cell/ReadVariableOpReadVariableOp,gru_while_gru_cell_readvariableop_resource_0*
_output_shapes

:`*
dtype02#
!gru/while/gru_cell/ReadVariableOp£
gru/while/gru_cell/unstackUnpack)gru/while/gru_cell/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num2
gru/while/gru_cell/unstackÈ
(gru/while/gru_cell/MatMul/ReadVariableOpReadVariableOp3gru_while_gru_cell_matmul_readvariableop_resource_0*
_output_shapes

:`*
dtype02*
(gru/while/gru_cell/MatMul/ReadVariableOpÚ
gru/while/gru_cell/MatMulMatMul4gru/while/TensorArrayV2Read/TensorListGetItem:item:00gru/while/gru_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
gru/while/gru_cell/MatMul¿
gru/while/gru_cell/BiasAddBiasAdd#gru/while/gru_cell/MatMul:product:0#gru/while/gru_cell/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
gru/while/gru_cell/BiasAddv
gru/while/gru_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru/while/gru_cell/Const
"gru/while/gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2$
"gru/while/gru_cell/split/split_dimø
gru/while/gru_cell/splitSplit+gru/while/gru_cell/split/split_dim:output:0#gru/while/gru_cell/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
gru/while/gru_cell/splitÎ
*gru/while/gru_cell/MatMul_1/ReadVariableOpReadVariableOp5gru_while_gru_cell_matmul_1_readvariableop_resource_0*
_output_shapes

: `*
dtype02,
*gru/while/gru_cell/MatMul_1/ReadVariableOpÃ
gru/while/gru_cell/MatMul_1MatMulgru_while_placeholder_22gru/while/gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
gru/while/gru_cell/MatMul_1Å
gru/while/gru_cell/BiasAdd_1BiasAdd%gru/while/gru_cell/MatMul_1:product:0#gru/while/gru_cell/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
gru/while/gru_cell/BiasAdd_1
gru/while/gru_cell/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"        ÿÿÿÿ2
gru/while/gru_cell/Const_1
$gru/while/gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2&
$gru/while/gru_cell/split_1/split_dim²
gru/while/gru_cell/split_1SplitV%gru/while/gru_cell/BiasAdd_1:output:0#gru/while/gru_cell/Const_1:output:0-gru/while/gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
gru/while/gru_cell/split_1³
gru/while/gru_cell/addAddV2!gru/while/gru_cell/split:output:0#gru/while/gru_cell/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/while/gru_cell/add
gru/while/gru_cell/SigmoidSigmoidgru/while/gru_cell/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/while/gru_cell/Sigmoid·
gru/while/gru_cell/add_1AddV2!gru/while/gru_cell/split:output:1#gru/while/gru_cell/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/while/gru_cell/add_1
gru/while/gru_cell/Sigmoid_1Sigmoidgru/while/gru_cell/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/while/gru_cell/Sigmoid_1°
gru/while/gru_cell/mulMul gru/while/gru_cell/Sigmoid_1:y:0#gru/while/gru_cell/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/while/gru_cell/mul®
gru/while/gru_cell/add_2AddV2!gru/while/gru_cell/split:output:2gru/while/gru_cell/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/while/gru_cell/add_2
gru/while/gru_cell/ReluRelugru/while/gru_cell/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/while/gru_cell/Relu¦
gru/while/gru_cell/mul_1Mulgru/while/gru_cell/Sigmoid:y:0gru_while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/while/gru_cell/mul_1y
gru/while/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru/while/gru_cell/sub/x¬
gru/while/gru_cell/subSub!gru/while/gru_cell/sub/x:output:0gru/while/gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/while/gru_cell/sub°
gru/while/gru_cell/mul_2Mulgru/while/gru_cell/sub:z:0%gru/while/gru_cell/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/while/gru_cell/mul_2«
gru/while/gru_cell/add_3AddV2gru/while/gru_cell/mul_1:z:0gru/while/gru_cell/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/while/gru_cell/add_3ð
.gru/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemgru_while_placeholder_1gru_while_placeholdergru/while/gru_cell/add_3:z:0*
_output_shapes
: *
element_dtype020
.gru/while/TensorArrayV2Write/TensorListSetItemd
gru/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
gru/while/add/yy
gru/while/addAddV2gru_while_placeholdergru/while/add/y:output:0*
T0*
_output_shapes
: 2
gru/while/addh
gru/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
gru/while/add_1/y
gru/while/add_1AddV2 gru_while_gru_while_loop_countergru/while/add_1/y:output:0*
T0*
_output_shapes
: 2
gru/while/add_1æ
gru/while/IdentityIdentitygru/while/add_1:z:0)^gru/while/gru_cell/MatMul/ReadVariableOp+^gru/while/gru_cell/MatMul_1/ReadVariableOp"^gru/while/gru_cell/ReadVariableOp*
T0*
_output_shapes
: 2
gru/while/Identityý
gru/while/Identity_1Identity&gru_while_gru_while_maximum_iterations)^gru/while/gru_cell/MatMul/ReadVariableOp+^gru/while/gru_cell/MatMul_1/ReadVariableOp"^gru/while/gru_cell/ReadVariableOp*
T0*
_output_shapes
: 2
gru/while/Identity_1è
gru/while/Identity_2Identitygru/while/add:z:0)^gru/while/gru_cell/MatMul/ReadVariableOp+^gru/while/gru_cell/MatMul_1/ReadVariableOp"^gru/while/gru_cell/ReadVariableOp*
T0*
_output_shapes
: 2
gru/while/Identity_2
gru/while/Identity_3Identity>gru/while/TensorArrayV2Write/TensorListSetItem:output_handle:0)^gru/while/gru_cell/MatMul/ReadVariableOp+^gru/while/gru_cell/MatMul_1/ReadVariableOp"^gru/while/gru_cell/ReadVariableOp*
T0*
_output_shapes
: 2
gru/while/Identity_3
gru/while/Identity_4Identitygru/while/gru_cell/add_3:z:0)^gru/while/gru_cell/MatMul/ReadVariableOp+^gru/while/gru_cell/MatMul_1/ReadVariableOp"^gru/while/gru_cell/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru/while/Identity_4"l
3gru_while_gru_cell_matmul_1_readvariableop_resource5gru_while_gru_cell_matmul_1_readvariableop_resource_0"h
1gru_while_gru_cell_matmul_readvariableop_resource3gru_while_gru_cell_matmul_readvariableop_resource_0"Z
*gru_while_gru_cell_readvariableop_resource,gru_while_gru_cell_readvariableop_resource_0"@
gru_while_gru_strided_slice_1gru_while_gru_strided_slice_1_0"1
gru_while_identitygru/while/Identity:output:0"5
gru_while_identity_1gru/while/Identity_1:output:0"5
gru_while_identity_2gru/while/Identity_2:output:0"5
gru_while_identity_3gru/while/Identity_3:output:0"5
gru_while_identity_4gru/while/Identity_4:output:0"¸
Ygru_while_tensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensor[gru_while_tensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :ÿÿÿÿÿÿÿÿÿ : : :::2T
(gru/while/gru_cell/MatMul/ReadVariableOp(gru/while/gru_cell/MatMul/ReadVariableOp2X
*gru/while/gru_cell/MatMul_1/ReadVariableOp*gru/while/gru_cell/MatMul_1/ReadVariableOp2F
!gru/while/gru_cell/ReadVariableOp!gru/while/gru_cell/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ :

_output_shapes
: :

_output_shapes
: 
Ë
¥
while_cond_10742
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_10742___redundant_placeholder03
/while_while_cond_10742___redundant_placeholder13
/while_while_cond_10742___redundant_placeholder23
/while_while_cond_10742___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
óF
¤
while_body_11083
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0.
*while_gru_cell_1_readvariableop_resource_05
1while_gru_cell_1_matmul_readvariableop_resource_07
3while_gru_cell_1_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor,
(while_gru_cell_1_readvariableop_resource3
/while_gru_cell_1_matmul_readvariableop_resource5
1while_gru_cell_1_matmul_1_readvariableop_resource¢&while/gru_cell_1/MatMul/ReadVariableOp¢(while/gru_cell_1/MatMul_1/ReadVariableOp¢while/gru_cell_1/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÓ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem­
while/gru_cell_1/ReadVariableOpReadVariableOp*while_gru_cell_1_readvariableop_resource_0*
_output_shapes

:0*
dtype02!
while/gru_cell_1/ReadVariableOp
while/gru_cell_1/unstackUnpack'while/gru_cell_1/ReadVariableOp:value:0*
T0* 
_output_shapes
:0:0*	
num2
while/gru_cell_1/unstackÂ
&while/gru_cell_1/MatMul/ReadVariableOpReadVariableOp1while_gru_cell_1_matmul_readvariableop_resource_0*
_output_shapes

: 0*
dtype02(
&while/gru_cell_1/MatMul/ReadVariableOpÐ
while/gru_cell_1/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/gru_cell_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02
while/gru_cell_1/MatMul·
while/gru_cell_1/BiasAddBiasAdd!while/gru_cell_1/MatMul:product:0!while/gru_cell_1/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02
while/gru_cell_1/BiasAddr
while/gru_cell_1/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/gru_cell_1/Const
 while/gru_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2"
 while/gru_cell_1/split/split_dimð
while/gru_cell_1/splitSplit)while/gru_cell_1/split/split_dim:output:0!while/gru_cell_1/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
while/gru_cell_1/splitÈ
(while/gru_cell_1/MatMul_1/ReadVariableOpReadVariableOp3while_gru_cell_1_matmul_1_readvariableop_resource_0*
_output_shapes

:0*
dtype02*
(while/gru_cell_1/MatMul_1/ReadVariableOp¹
while/gru_cell_1/MatMul_1MatMulwhile_placeholder_20while/gru_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02
while/gru_cell_1/MatMul_1½
while/gru_cell_1/BiasAdd_1BiasAdd#while/gru_cell_1/MatMul_1:product:0!while/gru_cell_1/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02
while/gru_cell_1/BiasAdd_1
while/gru_cell_1/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      ÿÿÿÿ2
while/gru_cell_1/Const_1
"while/gru_cell_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2$
"while/gru_cell_1/split_1/split_dim¨
while/gru_cell_1/split_1SplitV#while/gru_cell_1/BiasAdd_1:output:0!while/gru_cell_1/Const_1:output:0+while/gru_cell_1/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
while/gru_cell_1/split_1«
while/gru_cell_1/addAddV2while/gru_cell_1/split:output:0!while/gru_cell_1/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_1/add
while/gru_cell_1/SigmoidSigmoidwhile/gru_cell_1/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_1/Sigmoid¯
while/gru_cell_1/add_1AddV2while/gru_cell_1/split:output:1!while/gru_cell_1/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_1/add_1
while/gru_cell_1/Sigmoid_1Sigmoidwhile/gru_cell_1/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_1/Sigmoid_1¨
while/gru_cell_1/mulMulwhile/gru_cell_1/Sigmoid_1:y:0!while/gru_cell_1/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_1/mul¦
while/gru_cell_1/add_2AddV2while/gru_cell_1/split:output:2while/gru_cell_1/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_1/add_2
while/gru_cell_1/ReluReluwhile/gru_cell_1/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_1/Relu
while/gru_cell_1/mul_1Mulwhile/gru_cell_1/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_1/mul_1u
while/gru_cell_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
while/gru_cell_1/sub/x¤
while/gru_cell_1/subSubwhile/gru_cell_1/sub/x:output:0while/gru_cell_1/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_1/sub¨
while/gru_cell_1/mul_2Mulwhile/gru_cell_1/sub:z:0#while/gru_cell_1/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_1/mul_2£
while/gru_cell_1/add_3AddV2while/gru_cell_1/mul_1:z:0while/gru_cell_1/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_1/add_3Þ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_1/add_3:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1Ô
while/IdentityIdentitywhile/add_1:z:0'^while/gru_cell_1/MatMul/ReadVariableOp)^while/gru_cell_1/MatMul_1/ReadVariableOp ^while/gru_cell_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identityç
while/Identity_1Identitywhile_while_maximum_iterations'^while/gru_cell_1/MatMul/ReadVariableOp)^while/gru_cell_1/MatMul_1/ReadVariableOp ^while/gru_cell_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1Ö
while/Identity_2Identitywhile/add:z:0'^while/gru_cell_1/MatMul/ReadVariableOp)^while/gru_cell_1/MatMul_1/ReadVariableOp ^while/gru_cell_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0'^while/gru_cell_1/MatMul/ReadVariableOp)^while/gru_cell_1/MatMul_1/ReadVariableOp ^while/gru_cell_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3ô
while/Identity_4Identitywhile/gru_cell_1/add_3:z:0'^while/gru_cell_1/MatMul/ReadVariableOp)^while/gru_cell_1/MatMul_1/ReadVariableOp ^while/gru_cell_1/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_4"h
1while_gru_cell_1_matmul_1_readvariableop_resource3while_gru_cell_1_matmul_1_readvariableop_resource_0"d
/while_gru_cell_1_matmul_readvariableop_resource1while_gru_cell_1_matmul_readvariableop_resource_0"V
(while_gru_cell_1_readvariableop_resource*while_gru_cell_1_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :ÿÿÿÿÿÿÿÿÿ: : :::2P
&while/gru_cell_1/MatMul/ReadVariableOp&while/gru_cell_1/MatMul/ReadVariableOp2T
(while/gru_cell_1/MatMul_1/ReadVariableOp(while/gru_cell_1/MatMul_1/ReadVariableOp2B
while/gru_cell_1/ReadVariableOpwhile/gru_cell_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
Y
Ç
>__inference_gru_layer_call_and_return_conditional_losses_10126
inputs_0$
 gru_cell_readvariableop_resource+
'gru_cell_matmul_readvariableop_resource-
)gru_cell_matmul_1_readvariableop_resource
identity¢gru_cell/MatMul/ReadVariableOp¢ gru_cell/MatMul_1/ReadVariableOp¢gru_cell/ReadVariableOp¢whileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ü
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_2
gru_cell/ReadVariableOpReadVariableOp gru_cell_readvariableop_resource*
_output_shapes

:`*
dtype02
gru_cell/ReadVariableOp
gru_cell/unstackUnpackgru_cell/ReadVariableOp:value:0*
T0* 
_output_shapes
:`:`*	
num2
gru_cell/unstack¨
gru_cell/MatMul/ReadVariableOpReadVariableOp'gru_cell_matmul_readvariableop_resource*
_output_shapes

:`*
dtype02 
gru_cell/MatMul/ReadVariableOp 
gru_cell/MatMulMatMulstrided_slice_2:output:0&gru_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
gru_cell/MatMul
gru_cell/BiasAddBiasAddgru_cell/MatMul:product:0gru_cell/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
gru_cell/BiasAddb
gru_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru_cell/Const
gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
gru_cell/split/split_dimÐ
gru_cell/splitSplit!gru_cell/split/split_dim:output:0gru_cell/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
gru_cell/split®
 gru_cell/MatMul_1/ReadVariableOpReadVariableOp)gru_cell_matmul_1_readvariableop_resource*
_output_shapes

: `*
dtype02"
 gru_cell/MatMul_1/ReadVariableOp
gru_cell/MatMul_1MatMulzeros:output:0(gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
gru_cell/MatMul_1
gru_cell/BiasAdd_1BiasAddgru_cell/MatMul_1:product:0gru_cell/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
gru_cell/BiasAdd_1y
gru_cell/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"        ÿÿÿÿ2
gru_cell/Const_1
gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
gru_cell/split_1/split_dim
gru_cell/split_1SplitVgru_cell/BiasAdd_1:output:0gru_cell/Const_1:output:0#gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ *
	num_split2
gru_cell/split_1
gru_cell/addAddV2gru_cell/split:output:0gru_cell/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/adds
gru_cell/SigmoidSigmoidgru_cell/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/Sigmoid
gru_cell/add_1AddV2gru_cell/split:output:1gru_cell/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/add_1y
gru_cell/Sigmoid_1Sigmoidgru_cell/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/Sigmoid_1
gru_cell/mulMulgru_cell/Sigmoid_1:y:0gru_cell/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/mul
gru_cell/add_2AddV2gru_cell/split:output:2gru_cell/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/add_2l
gru_cell/ReluRelugru_cell/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/Relu
gru_cell/mul_1Mulgru_cell/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/mul_1e
gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru_cell/sub/x
gru_cell/subSubgru_cell/sub/x:output:0gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/sub
gru_cell/mul_2Mulgru_cell/sub:z:0gru_cell/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/mul_2
gru_cell/add_3AddV2gru_cell/mul_1:z:0gru_cell/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
gru_cell/add_3
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    2
TensorArrayV2_1/element_shape¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0 gru_cell_readvariableop_resource'gru_cell_matmul_readvariableop_resource)gru_cell_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ : : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_10036*
condR
while_cond_10035*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ : : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    22
0TensorArrayV2Stack/TensorListStack/element_shapeñ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm®
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeÖ
IdentityIdentitytranspose_1:y:0^gru_cell/MatMul/ReadVariableOp!^gru_cell/MatMul_1/ReadVariableOp^gru_cell/ReadVariableOp^while*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::2@
gru_cell/MatMul/ReadVariableOpgru_cell/MatMul/ReadVariableOp2D
 gru_cell/MatMul_1/ReadVariableOp gru_cell/MatMul_1/ReadVariableOp22
gru_cell/ReadVariableOpgru_cell/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0

`
A__inference_dropout_layer_call_and_return_conditional_losses_8537

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
dropout/Constw
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape¸
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dropout/GreaterEqual/yÂ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dropout/Cast~
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dropout/Mul_1i
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs"±L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*°
serving_default
C
	gru_input6
serving_default_gru_input:0ÿÿÿÿÿÿÿÿÿ9
dense0
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:Â
Û3
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
	optimizer
	variables
trainable_variables
	regularization_losses

	keras_api

signatures
*x&call_and_return_all_conditional_losses
y__call__
z_default_save_signature"1
_tf_keras_sequentialâ0{"class_name": "Sequential", "name": "sequential", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 4, 8]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "gru_input"}}, {"class_name": "GRU", "config": {"name": "gru", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 4, 8]}, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 32, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2, "reset_after": true}}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0, "noise_shape": null, "seed": null}}, {"class_name": "GRU", "config": {"name": "gru_1", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 16, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2, "reset_after": true}}, {"class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 3, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 8]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 4, 8]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 4, 8]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "gru_input"}}, {"class_name": "GRU", "config": {"name": "gru", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 4, 8]}, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 32, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2, "reset_after": true}}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0, "noise_shape": null, "seed": null}}, {"class_name": "GRU", "config": {"name": "gru_1", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 16, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2, "reset_after": true}}, {"class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 3, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": "mean_squared_error", "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
¤
cell

state_spec
	variables
trainable_variables
regularization_losses
	keras_api
*{&call_and_return_all_conditional_losses
|__call__"û

_tf_keras_rnn_layerÝ
{"class_name": "GRU", "name": "gru", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 4, 8]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "gru", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 4, 8]}, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 32, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2, "reset_after": true}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 8]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 4, 8]}}
ß
	variables
trainable_variables
regularization_losses
	keras_api
*}&call_and_return_all_conditional_losses
~__call__"Ð
_tf_keras_layer¶{"class_name": "Dropout", "name": "dropout", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0, "noise_shape": null, "seed": null}}
µ
cell

state_spec
	variables
trainable_variables
regularization_losses
	keras_api
*&call_and_return_all_conditional_losses
__call__"

_tf_keras_rnn_layerí	{"class_name": "GRU", "name": "gru_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "gru_1", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 16, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2, "reset_after": true}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 32]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 4, 32]}}
å
	variables
trainable_variables
regularization_losses
	keras_api
+&call_and_return_all_conditional_losses
__call__"Ô
_tf_keras_layerº{"class_name": "Dropout", "name": "dropout_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0, "noise_shape": null, "seed": null}}
ï

 kernel
!bias
"	variables
#trainable_variables
$regularization_losses
%	keras_api
+&call_and_return_all_conditional_losses
__call__"È
_tf_keras_layer®{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 3, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16]}}
ã
&iter

'beta_1

(beta_2
	)decay
*learning_rate mh!mi+mj,mk-ml.mm/mn0mo vp!vq+vr,vs-vt.vu/vv0vw"
	optimizer
X
+0
,1
-2
.3
/4
05
 6
!7"
trackable_list_wrapper
X
+0
,1
-2
.3
/4
05
 6
!7"
trackable_list_wrapper
 "
trackable_list_wrapper
Ê
	variables
trainable_variables

1layers
2layer_regularization_losses
3non_trainable_variables
4layer_metrics
5metrics
	regularization_losses
y__call__
z_default_save_signature
*x&call_and_return_all_conditional_losses
&x"call_and_return_conditional_losses"
_generic_user_object
-
serving_default"
signature_map


+kernel
,recurrent_kernel
-bias
6	variables
7trainable_variables
8regularization_losses
9	keras_api
+&call_and_return_all_conditional_losses
__call__"á
_tf_keras_layerÇ{"class_name": "GRUCell", "name": "gru_cell", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "gru_cell", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2, "reset_after": true}}
 "
trackable_list_wrapper
5
+0
,1
-2"
trackable_list_wrapper
5
+0
,1
-2"
trackable_list_wrapper
 "
trackable_list_wrapper
¹
	variables
trainable_variables

:layers
;layer_regularization_losses

<states
=non_trainable_variables
>layer_metrics
?metrics
regularization_losses
|__call__
*{&call_and_return_all_conditional_losses
&{"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
	variables
trainable_variables

@layers
Alayer_regularization_losses
Bnon_trainable_variables
Clayer_metrics
Dmetrics
regularization_losses
~__call__
*}&call_and_return_all_conditional_losses
&}"call_and_return_conditional_losses"
_generic_user_object
¢

.kernel
/recurrent_kernel
0bias
E	variables
Ftrainable_variables
Gregularization_losses
H	keras_api
+&call_and_return_all_conditional_losses
__call__"å
_tf_keras_layerË{"class_name": "GRUCell", "name": "gru_cell_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "gru_cell_1", "trainable": true, "dtype": "float32", "units": 16, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2, "reset_after": true}}
 "
trackable_list_wrapper
5
.0
/1
02"
trackable_list_wrapper
5
.0
/1
02"
trackable_list_wrapper
 "
trackable_list_wrapper
º
	variables
trainable_variables

Ilayers
Jlayer_regularization_losses

Kstates
Lnon_trainable_variables
Mlayer_metrics
Nmetrics
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°
	variables
trainable_variables

Olayers
Player_regularization_losses
Qnon_trainable_variables
Rlayer_metrics
Smetrics
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
:2dense/kernel
:2
dense/bias
.
 0
!1"
trackable_list_wrapper
.
 0
!1"
trackable_list_wrapper
 "
trackable_list_wrapper
°
"	variables
#trainable_variables

Tlayers
Ulayer_regularization_losses
Vnon_trainable_variables
Wlayer_metrics
Xmetrics
$regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
%:#`2gru/gru_cell/kernel
/:- `2gru/gru_cell/recurrent_kernel
#:!`2gru/gru_cell/bias
):' 02gru_1/gru_cell_1/kernel
3:102!gru_1/gru_cell_1/recurrent_kernel
':%02gru_1/gru_cell_1/bias
C
0
1
2
3
4"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
'
Y0"
trackable_list_wrapper
5
+0
,1
-2"
trackable_list_wrapper
5
+0
,1
-2"
trackable_list_wrapper
 "
trackable_list_wrapper
°
6	variables
7trainable_variables

Zlayers
[layer_regularization_losses
\non_trainable_variables
]layer_metrics
^metrics
8regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
5
.0
/1
02"
trackable_list_wrapper
5
.0
/1
02"
trackable_list_wrapper
 "
trackable_list_wrapper
°
E	variables
Ftrainable_variables

_layers
`layer_regularization_losses
anon_trainable_variables
blayer_metrics
cmetrics
Gregularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
»
	dtotal
	ecount
f	variables
g	keras_api"
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
:  (2total
:  (2count
.
d0
e1"
trackable_list_wrapper
-
f	variables"
_generic_user_object
#:!2Adam/dense/kernel/m
:2Adam/dense/bias/m
*:(`2Adam/gru/gru_cell/kernel/m
4:2 `2$Adam/gru/gru_cell/recurrent_kernel/m
(:&`2Adam/gru/gru_cell/bias/m
.:, 02Adam/gru_1/gru_cell_1/kernel/m
8:602(Adam/gru_1/gru_cell_1/recurrent_kernel/m
,:*02Adam/gru_1/gru_cell_1/bias/m
#:!2Adam/dense/kernel/v
:2Adam/dense/bias/v
*:(`2Adam/gru/gru_cell/kernel/v
4:2 `2$Adam/gru/gru_cell/recurrent_kernel/v
(:&`2Adam/gru/gru_cell/bias/v
.:, 02Adam/gru_1/gru_cell_1/kernel/v
8:602(Adam/gru_1/gru_cell_1/recurrent_kernel/v
,:*02Adam/gru_1/gru_cell_1/bias/v
Þ2Û
D__inference_sequential_layer_call_and_return_conditional_losses_9444
D__inference_sequential_layer_call_and_return_conditional_losses_8984
D__inference_sequential_layer_call_and_return_conditional_losses_8959
D__inference_sequential_layer_call_and_return_conditional_losses_9766À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ò2ï
)__inference_sequential_layer_call_fn_9787
)__inference_sequential_layer_call_fn_9808
)__inference_sequential_layer_call_fn_9077
)__inference_sequential_layer_call_fn_9031À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ã2à
__inference__wrapped_model_7050¼
²
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *,¢)
'$
	gru_inputÿÿÿÿÿÿÿÿÿ
Ú2×
=__inference_gru_layer_call_and_return_conditional_losses_9967
>__inference_gru_layer_call_and_return_conditional_losses_10307
>__inference_gru_layer_call_and_return_conditional_losses_10466
>__inference_gru_layer_call_and_return_conditional_losses_10126Õ
Ì²È
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ï2ì
#__inference_gru_layer_call_fn_10488
#__inference_gru_layer_call_fn_10477
#__inference_gru_layer_call_fn_10137
#__inference_gru_layer_call_fn_10148Õ
Ì²È
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Â2¿
B__inference_dropout_layer_call_and_return_conditional_losses_10505
B__inference_dropout_layer_call_and_return_conditional_losses_10500´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
'__inference_dropout_layer_call_fn_10515
'__inference_dropout_layer_call_fn_10510´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ã2à
@__inference_gru_1_layer_call_and_return_conditional_losses_10833
@__inference_gru_1_layer_call_and_return_conditional_losses_11014
@__inference_gru_1_layer_call_and_return_conditional_losses_11173
@__inference_gru_1_layer_call_and_return_conditional_losses_10674Õ
Ì²È
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
÷2ô
%__inference_gru_1_layer_call_fn_11184
%__inference_gru_1_layer_call_fn_10844
%__inference_gru_1_layer_call_fn_10855
%__inference_gru_1_layer_call_fn_11195Õ
Ì²È
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Æ2Ã
D__inference_dropout_1_layer_call_and_return_conditional_losses_11207
D__inference_dropout_1_layer_call_and_return_conditional_losses_11212´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
)__inference_dropout_1_layer_call_fn_11222
)__inference_dropout_1_layer_call_fn_11217´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ê2ç
@__inference_dense_layer_call_and_return_conditional_losses_11232¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ï2Ì
%__inference_dense_layer_call_fn_11241¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ËBÈ
"__inference_signature_wrapper_9108	gru_input"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Î2Ë
C__inference_gru_cell_layer_call_and_return_conditional_losses_11281
C__inference_gru_cell_layer_call_and_return_conditional_losses_11321¾
µ²±
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
(__inference_gru_cell_layer_call_fn_11335
(__inference_gru_cell_layer_call_fn_11349¾
µ²±
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ò2Ï
E__inference_gru_cell_1_layer_call_and_return_conditional_losses_11389
E__inference_gru_cell_1_layer_call_and_return_conditional_losses_11429¾
µ²±
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
*__inference_gru_cell_1_layer_call_fn_11443
*__inference_gru_cell_1_layer_call_fn_11457¾
µ²±
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
__inference__wrapped_model_7050q-+,0./ !6¢3
,¢)
'$
	gru_inputÿÿÿÿÿÿÿÿÿ
ª "-ª*
(
dense
denseÿÿÿÿÿÿÿÿÿ 
@__inference_dense_layer_call_and_return_conditional_losses_11232\ !/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 x
%__inference_dense_layer_call_fn_11241O !/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¤
D__inference_dropout_1_layer_call_and_return_conditional_losses_11207\3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¤
D__inference_dropout_1_layer_call_and_return_conditional_losses_11212\3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 |
)__inference_dropout_1_layer_call_fn_11217O3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿ|
)__inference_dropout_1_layer_call_fn_11222O3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿª
B__inference_dropout_layer_call_and_return_conditional_losses_10500d7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿ 
p
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ 
 ª
B__inference_dropout_layer_call_and_return_conditional_losses_10505d7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿ 
p 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ 
 
'__inference_dropout_layer_call_fn_10510W7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿ 
p
ª "ÿÿÿÿÿÿÿÿÿ 
'__inference_dropout_layer_call_fn_10515W7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿ 
p 
ª "ÿÿÿÿÿÿÿÿÿ ±
@__inference_gru_1_layer_call_and_return_conditional_losses_10674m0./?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿ 

 
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ±
@__inference_gru_1_layer_call_and_return_conditional_losses_10833m0./?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿ 

 
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Á
@__inference_gru_1_layer_call_and_return_conditional_losses_11014}0./O¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 

 
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Á
@__inference_gru_1_layer_call_and_return_conditional_losses_11173}0./O¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 

 
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
%__inference_gru_1_layer_call_fn_10844`0./?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿ 

 
p

 
ª "ÿÿÿÿÿÿÿÿÿ
%__inference_gru_1_layer_call_fn_10855`0./?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿ 

 
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
%__inference_gru_1_layer_call_fn_11184p0./O¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 

 
p

 
ª "ÿÿÿÿÿÿÿÿÿ
%__inference_gru_1_layer_call_fn_11195p0./O¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 

 
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
E__inference_gru_cell_1_layer_call_and_return_conditional_losses_11389·0./\¢Y
R¢O
 
inputsÿÿÿÿÿÿÿÿÿ 
'¢$
"
states/0ÿÿÿÿÿÿÿÿÿ
p
ª "R¢O
H¢E

0/0ÿÿÿÿÿÿÿÿÿ
$!

0/1/0ÿÿÿÿÿÿÿÿÿ
 
E__inference_gru_cell_1_layer_call_and_return_conditional_losses_11429·0./\¢Y
R¢O
 
inputsÿÿÿÿÿÿÿÿÿ 
'¢$
"
states/0ÿÿÿÿÿÿÿÿÿ
p 
ª "R¢O
H¢E

0/0ÿÿÿÿÿÿÿÿÿ
$!

0/1/0ÿÿÿÿÿÿÿÿÿ
 Ø
*__inference_gru_cell_1_layer_call_fn_11443©0./\¢Y
R¢O
 
inputsÿÿÿÿÿÿÿÿÿ 
'¢$
"
states/0ÿÿÿÿÿÿÿÿÿ
p
ª "D¢A

0ÿÿÿÿÿÿÿÿÿ
"

1/0ÿÿÿÿÿÿÿÿÿØ
*__inference_gru_cell_1_layer_call_fn_11457©0./\¢Y
R¢O
 
inputsÿÿÿÿÿÿÿÿÿ 
'¢$
"
states/0ÿÿÿÿÿÿÿÿÿ
p 
ª "D¢A

0ÿÿÿÿÿÿÿÿÿ
"

1/0ÿÿÿÿÿÿÿÿÿÿ
C__inference_gru_cell_layer_call_and_return_conditional_losses_11281·-+,\¢Y
R¢O
 
inputsÿÿÿÿÿÿÿÿÿ
'¢$
"
states/0ÿÿÿÿÿÿÿÿÿ 
p
ª "R¢O
H¢E

0/0ÿÿÿÿÿÿÿÿÿ 
$!

0/1/0ÿÿÿÿÿÿÿÿÿ 
 ÿ
C__inference_gru_cell_layer_call_and_return_conditional_losses_11321·-+,\¢Y
R¢O
 
inputsÿÿÿÿÿÿÿÿÿ
'¢$
"
states/0ÿÿÿÿÿÿÿÿÿ 
p 
ª "R¢O
H¢E

0/0ÿÿÿÿÿÿÿÿÿ 
$!

0/1/0ÿÿÿÿÿÿÿÿÿ 
 Ö
(__inference_gru_cell_layer_call_fn_11335©-+,\¢Y
R¢O
 
inputsÿÿÿÿÿÿÿÿÿ
'¢$
"
states/0ÿÿÿÿÿÿÿÿÿ 
p
ª "D¢A

0ÿÿÿÿÿÿÿÿÿ 
"

1/0ÿÿÿÿÿÿÿÿÿ Ö
(__inference_gru_cell_layer_call_fn_11349©-+,\¢Y
R¢O
 
inputsÿÿÿÿÿÿÿÿÿ
'¢$
"
states/0ÿÿÿÿÿÿÿÿÿ 
p 
ª "D¢A

0ÿÿÿÿÿÿÿÿÿ 
"

1/0ÿÿÿÿÿÿÿÿÿ Í
>__inference_gru_layer_call_and_return_conditional_losses_10126-+,O¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p 

 
ª "2¢/
(%
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 ³
>__inference_gru_layer_call_and_return_conditional_losses_10307q-+,?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿ

 
p

 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ 
 ³
>__inference_gru_layer_call_and_return_conditional_losses_10466q-+,?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿ

 
p 

 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ 
 Ì
=__inference_gru_layer_call_and_return_conditional_losses_9967-+,O¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p

 
ª "2¢/
(%
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 ¤
#__inference_gru_layer_call_fn_10137}-+,O¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p

 
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ¤
#__inference_gru_layer_call_fn_10148}-+,O¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p 

 
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
#__inference_gru_layer_call_fn_10477d-+,?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿ

 
p

 
ª "ÿÿÿÿÿÿÿÿÿ 
#__inference_gru_layer_call_fn_10488d-+,?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿ

 
p 

 
ª "ÿÿÿÿÿÿÿÿÿ ¹
D__inference_sequential_layer_call_and_return_conditional_losses_8959q-+,0./ !>¢;
4¢1
'$
	gru_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¹
D__inference_sequential_layer_call_and_return_conditional_losses_8984q-+,0./ !>¢;
4¢1
'$
	gru_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¶
D__inference_sequential_layer_call_and_return_conditional_losses_9444n-+,0./ !;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¶
D__inference_sequential_layer_call_and_return_conditional_losses_9766n-+,0./ !;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
)__inference_sequential_layer_call_fn_9031d-+,0./ !>¢;
4¢1
'$
	gru_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ
)__inference_sequential_layer_call_fn_9077d-+,0./ !>¢;
4¢1
'$
	gru_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
)__inference_sequential_layer_call_fn_9787a-+,0./ !;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ
)__inference_sequential_layer_call_fn_9808a-+,0./ !;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ¤
"__inference_signature_wrapper_9108~-+,0./ !C¢@
¢ 
9ª6
4
	gru_input'$
	gru_inputÿÿÿÿÿÿÿÿÿ"-ª*
(
dense
denseÿÿÿÿÿÿÿÿÿ