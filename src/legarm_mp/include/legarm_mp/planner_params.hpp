/* ----------------------------------------------------------------------------
 * Copyright (c) 2021, University of Leeds and Harbin Institute of Technology.
 * All Rights Reserved
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 *  @file   planner_params.hpp
 *  @author Jun Li (junlileeds@gmail.com)
 *  @brief  Header file for parameters
 *  @date   April 16, 2022
 **/

#pragma once

namespace legarm_mp {

/*! Available string variables used by the planner */
enum PlannerStringParam {
    PlannerStringParam_BaseName, 
};

enum PlannerStringVectorParam {
    PlannerStringVectorParam_LegNames,
    PlannerStringVectorParam_ArmNames,
    PlannerStringVectorParam_LegEndEffNames,
    PlannerStringVectorParam_ArmEndEffNames,
};

/*! Available int variables used by the planner */
enum PlannerIntParam {
    PlannerIntParam_NumLegs,
    PlannerIntParam_NumArms,
    PlannerIntParam_NumJoints,
};

/*! Available double variables used by the planner */
enum PlannerDoubleParam {
    PlannerDoubleParam_TimeStep,
    PlannerDoubleParam_BodyMass,
    PlannerDoubleParam_ClearanceRatio,
    PlannerDoubleParam_FricCoef,
    PlannerDoubleParam_FootOffset,
    PlannerDoubleParam_FootDeltaxLimit,
    PlannerDoubleParam_FootDeltayLimit,
};

/*! Available vector variables used by the planner */
enum PlannerVectorParam {
    PlannerVectorParam_BodyKp,
    PlannerVectorParam_BodyKd,
    PlannerVectorParam_BodyMaxAcc,
    PlannerVectorParam_RaibertKp,
    PlannerVectorParam_BodyInitPos,
    PlannerVectorParam_BodyInitVel,
    PlannerVectorParam_JointInitPos,
    PlannerVectorParam_JointInitVel,
};

/*! Available matrix variables used by the planner */
enum PlannerMatrixParam {
    PlannerMatrixParam_BodyInertia,
    PlannerMatrixParam_HipRelPos,
    PlannerMatrixParam_FootInitRelPos,
    PlannerMatrixParam_GripperInitRelPos,
};

}  // end legarm_mp namespace