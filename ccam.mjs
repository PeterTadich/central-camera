//ccam = central-camera

//ECMAScript module

//npm install https://github.com/PeterTadich/singular-value-decomposition https://github.com/PeterTadich/pseudo-inverse https://github.com/PeterTadich/matrix-computations

// To do:
//   - setup defaults in setCamParameters()
//   - test mappingPixelToWorld_check()
//   - test image_jacobian_inverse_check()

import * as hlao from 'matrix-computations';
import * as svdcmp from 'singular-value-decomposition';
import * as pinv from 'pseudo-inverse';
//import * as hlao from '../matrix-computations/hlao.mjs';
//import * as svdcmp from '../singular-value-decomposition/svdcmp.mjs';
//import * as pinv from '../pseudo-inverse/pinv.mjs';
//import * as hlao from '../../node_modules/matrix-computations/hlao.mjs';
//import * as svdcmp from '../../node_modules/singular-value-decomposition/svdcmp.mjs';
//import * as pinv from '../../node_modules/pseudo-inverse/pinv.mjs';

var cameraParameters = {
    //Intrinsic parameters.
    pu: 10e-6, //Pixel size.
    pv: 10e-6, //Pixel size.
    f: 8.0/1000.0, //focal.
    resolution: 1024.0,
    principalPoint: { //[u0,v0] Principal point [resolution/2,resolution/2].
        u0: 1024.0/2.0,
        v0: 1024.0/2.0
    },
    //Extrinsic parameters.
    T: [
        [1.0,0.0,0.0,0.0],
        [0.0,1.0,0.0,0.0],
        [0.0,0.0,1.0,0.0],
        [0.0,0.0,0.0,1.0]
    ]
};

var point_XYZ = [[0.05],[0.0],[0.50]]; //Defined in the world coordinate frame.

function setCamParameters(opts){
    cameraParameters.pu = opts.pu;
    cameraParameters.pv = opts.pv;
    cameraParameters.f = opts.f;
    cameraParameters.principalPoint.u0 = opts.principalPoint.u0;
    cameraParameters.principalPoint.v0 = opts.principalPoint.v0;
}

//Central camera model. Mapping world coordinates to pixel coordinates (image plane).
function mappingWorldToPixel(camera,point_XYZ){
    //Intrinsic.
    var K = hlao.matrix_multiplication(
        [
            [1.0/camera.pu,           0.0, camera.principalPoint.u0],
            [          0.0, 1.0/camera.pv, camera.principalPoint.v0],
            [          0.0,           0.0,                      1.0]
        ],
        [
            [camera.f,      0.0, 0.0, 0.0],
            [     0.0, camera.f, 0.0, 0.0],
            [     0.0,      0.0, 1.0, 0.0]
        ]
    );
    //Camera matrix.
    var C = hlao.matrix_multiplication(K,camera.T);
    //Pixel coordinates in homogenous coordinates.
    var h = hlao.matrix_multiplication(C,point_XYZ.concat([[1]])); // 'point_XYZ' To homogenous coordinates.
    
    return h;
}

/*
var h = mappingWorldToPixel(cameraParameters,point_XYZ);
mappingPixelToWorld(cameraParameters,h);
*/
//Central camera model. Map the pixel coordinates to the world coordinate.
function mappingPixelToWorld(camera,h){
    var debug = 0;
    //[P;0] = pinv(cam.C) * h where h = cam.C*P gives h = [296.0, 256.0, 0.50] for P = [0.05; 0.0; 0.5; 1.0];
    //Intrinsic.
    var K = hlao.matrix_multiplication(
        [
            [1.0/camera.pu,           0.0, camera.principalPoint.u0],
            [          0.0, 1.0/camera.pv, camera.principalPoint.v0],
            [          0.0,           0.0,                      1.0]
        ],
        [
            [camera.f,      0.0, 0.0, 0.0],
            [     0.0, camera.f, 0.0, 0.0],
            [     0.0,      0.0, 1.0, 0.0]
        ]
    );
    //Camera matrix.
    var C = hlao.matrix_multiplication(K,camera.T);
    //Pseudo inverse (right inverse). A^-1 = A^T(AA^T)^-1
    //Step 1. AA^T (3x4)x(4x3) = 3x3
    var AAT = hlao.matrix_multiplication(C,hlao.matrix_transpose(C));
    //var dim = size(AAT); //3x3
    //var m = dim[0]; //Number of rows.
    //var n = dim[1]; //Number of columns.
    var m; var n;
    [m,n] = [AAT.length,AAT[0].length];
    if(debug) console.log('A x AT: ' + m + ' x ' + n);
    //Step 2. Adjust the 'AAT' matrix for SVD.
    for(var i=0;i<m;i=i+1){ //3x3 to 3x4. For each row of AAT[] add an extra element '0.0' to the beginning of the array (beginning of the row).
        AAT[i].unshift(0.0);
    }
    var offsetRow = []; //Create an array of zeroes (row vector) 1x4.
    for(var i=0;i<(n+1);i=i+1){ //Only 3 + 1 elements as AAT[] is 3 x 3 matrix or 3 x 4 matrix with padded zeroes.
        offsetRow.push(0.0);
    }
    AAT.unshift(offsetRow); //Add the row vector of zeroes to the beginning of AAT[] array - now a 4 x 4 matrix.
    //Step 3. SVD.
    var w = hlao.zeros_vector((n+1),'row'); //Row vector where index = 0 is undefined.
    var v = hlao.zeros_matrix((n+1),(n+1)); //Matrix.
    var uwv = svdcmp.svdcmp(AAT, m, n, w, v);
    var U = svdcmp.svdClean(uwv[0]); // Drop the first element in the array as it is zero.
    var S = svdcmp.svdClean(uwv[1]); //W
    var V = svdcmp.svdClean(uwv[2]);
    if(debug) console.log('U: ' + U);
    if(debug) console.log('S: ' + S);
    if(debug) console.log('V: ' + V);
    //ref:
    //   - http://www.kwon3d.com/theory/jkinem/svd.html
    //   - http://au.mathworks.com/help/matlab/ref/svd.html
    //   - https://au.mathworks.com/help/matlab/ref/pinv.html
    //   - https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_pseudoinverse
    var Sinv = [
        [1.0/S[0],     0.0,     0.0],
        [     0.0,1.0/S[1],     0.0],
        [     0.0,     0.0,1.0/S[2]]
    ];
    var AATinv = hlao.matrix_multiplication( //<--- IMPORTANT: different from I:\code\spatial_v2\js\AnomalyDetection\ad.js
            hlao.matrix_multiplication(
                V,Sinv
            ),
            hlao.matrix_transpose(U)
        );
    if(debug) console.log('(A x AT)-1: ' + AATinv);
    //Step 4. A^T(AA^T)^-1
    var Cinv = hlao.matrix_multiplication(hlao.matrix_transpose(C),AATinv);
    //Step 5. C-1 x h
    var point_XYZ = hlao.matrix_multiplication(Cinv,h);
    point_XYZ.pop(); // Drop the last element. 'point_XYZ' back to Cartesian coordinates.
    return point_XYZ;
}

function mappingPixelToWorld_check(camera,h){
    var debug = 0;
    //[P;0] = pinv(cam.C) * h where h = cam.C*P gives h = [296.0, 256.0, 0.50] for P = [0.05; 0.0; 0.5; 1.0];
    //Intrinsic.
    var K = hlao.matrix_multiplication(
        [
            [1.0/camera.pu,           0.0, camera.principalPoint.u0],
            [          0.0, 1.0/camera.pv, camera.principalPoint.v0],
            [          0.0,           0.0,                      1.0]
        ],
        [
            [camera.f,      0.0, 0.0, 0.0],
            [     0.0, camera.f, 0.0, 0.0],
            [     0.0,      0.0, 1.0, 0.0]
        ]
    );
    //Camera matrix.
    var C = hlao.matrix_multiplication(K,camera.T);
    //Pseudo inverse
    //Step 1, 2, 3 and 4.
    var Cinv = pinv.pinv(C);
    //Step 5. C-1 x h
    var point_XYZ = hlao.matrix_multiplication(Cinv,h);
    return point_XYZ;
}

/*
//Taken from rot3dfit.js
function svdClean(A){
    var dim = size(A);
    var m = dim[0]; //Rows.
    
    if(m > 1){
        for(var i=0;i<m;i=i+1){ //For each row of 'A' drop the extra element '0.0' at the beginning of the array (beginning of the row).
            A[i].shift();
        }
        A.shift(); //Drop the first row.
    } else {
        A.shift(); //It is a row vector hence just drop the first element.
    }
    
    return A;
}
*/

//ref: Robotics, Vision and Control. Page 461.
//MATLAB:
//cam = CentralCamera('default');
//J = cam.visjac_p([600 600]', 5);
// J =
// -160.0000         0   17.6000    9.6800 -809.6800   88.0000
//         0 -160.0000   17.6000  809.6800   -9.6800  -88.0000
/*
//JavaScript:
var J = image_jacobian(cameraParameters, 5, [[600], [600]]);
console.log(J);
//J = [
//  [-160,    0, 17.6,   9.68, -809.68,  88],
//  [   0, -160, 17.6, 809.68,   -9.68, -88]
//]
*/
function image_jacobian(camera, Z, pixel_coords){
    var f = camera.f;
    var rho_u = camera.pu;
    var rho_v = camera.pv;
    var u_bar = pixel_coords[0][0] - camera.principalPoint.u0;
    var v_bar = pixel_coords[1][0] - camera.principalPoint.v0;
    var Jp = [
        [-1.0*f/(rho_u*Z),              0.0, u_bar/Z,                     (rho_u*u_bar*v_bar)/f, -1.0*(f*f + rho_u*rho_u*u_bar*u_bar)/(rho_u*f),      v_bar],
        [             0.0, -1.0*f/(rho_v*Z), v_bar/Z, (f*f + rho_v*rho_v*v_bar*v_bar)/(rho_v*f),                       -1.0*rho_v*u_bar*v_bar/f, -1.0*u_bar]
    ];
    return Jp;
}

// Calculate the pseudo-inverse of the image Jacobian.
/*
//Note 'cameraParameters' = default parameters.
var J = image_jacobian(cameraParameters, 5, [[600], [600]]);
var Jstar = image_jacobian_inverse(J);
console.log(Jstar);
// Jstar = [
//     [ -0.00023214290105036034, 0.000002775347399179349 ],
//     [ 0.0000027753473991792644, -0.00023214290105036026 ],
//     [ 0.00002523043090162992, 0.000025230430901629903 ],
//     [ 1.3721933745519665e-19, 0.0011745912422476978 ],
//     [ -0.001174591242247698, 2.964615315390051e-19 ],
//     [ 0.00012920503664724678, -0.0001292050366472468 ],
// ];
*/
function image_jacobian_inverse(J){
    var debug = 0;
    
    // Pseudo inverse (right inverse). J* = J^T * (J*J^T)-1 (MATLAB pinv()).
    //    Part 1. J*J^T (2x6 x (2x6)^T = 2x6 x 6x2 = 2x2)
    var JJT = hlao.matrix_multiplication(J,hlao.matrix_transpose(J));
    //var dim = size(JJT); //2x2
    //var m = dim[0]; //Number of rows.
    //var n = dim[1]; //Number of columns.
    var m; var n;
    [m,n] = [JJT.length,JJT[0].length];
    if(debug) console.log('J x JT: ' + m + ' x ' + n);
    //    Part 2. Adjust the 'JJT' matrix for SVD.
    for(var i=0;i<m;i=i+1){ //2x2 to 2x3. For each row of JJT[] add an extra element '0.0' to the beginning of the array (beginning of the row).
        JJT[i].unshift(0.0);
    }
    var offsetRow = []; //Create an array of zeroes (row vector) 1x3.
    for(var i=0;i<(n+1);i=i+1){ //Only 2 + 1 elements as JJT[] is 2 x 2 matrix or 2 x 3 matrix with padded zeroes.
        offsetRow.push(0.0);
    }
    JJT.unshift(offsetRow); //Add the row vector of zeroes to the beginning of JJT[] array - now a 3 x 3 matrix.
    //    Part 3. SVD.
    var w = hlao.zeros_vector((n+1),'row'); //Row vector where index = 0 is undefined.
    var v = hlao.zeros_matrix((n+1),(n+1)); //Matrix.
    var uwv = svdcmp.svdcmp(JJT, m, n, w, v);
    var U = svdcmp.svdClean(uwv[0]); //Drop the first element in the array as it is zero.
    var S = svdcmp.svdClean(uwv[1]); //W
    var V = svdcmp.svdClean(uwv[2]);
    if(debug) console.log('U: ' + U);
    if(debug) console.log('S: ' + S);
    if(debug) console.log('V: ' + V);
    //ref:
    //   - http://www.kwon3d.com/theory/jkinem/svd.html
    //   - http://au.mathworks.com/help/matlab/ref/svd.html
    //   - https://au.mathworks.com/help/matlab/ref/pinv.html
    //   - https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_pseudoinverse
    var Sinv = [
        [1.0/S[0],     0.0],
        [     0.0,1.0/S[1]],
    ];
    var JJTinv = hlao.matrix_multiplication(
            hlao.matrix_multiplication(
                V,Sinv
            ),
            hlao.matrix_transpose(U)
        );
    if(debug) console.log('(J x JT)-1: ' + JJTinv);
    //Part 4.  J^T * (J*J^T)-1
    var Jstar = hlao.matrix_multiplication(hlao.matrix_transpose(J),JJTinv);
    if(debug) console.log('J*: ' + Jstar);
    
    return Jstar;
}

function image_jacobian_inverse_check(J){
    //Pseudo inverse
    //Step 1, 2, 3 and 4.
    var Jinv = pinv.pinv(J);
    return Jinv;
}

export {
    cameraParameters,
    point_XYZ,
    setCamParameters,
    mappingWorldToPixel,
    mappingPixelToWorld,
    mappingPixelToWorld_check,
    image_jacobian,
    image_jacobian_inverse,
    image_jacobian_inverse_check
};