import * as THREE from './three.module.js';
import { Line2 } from './lines/Line2.js';
import { LineMaterial } from './lines/LineMaterial.js';
import { LineGeometry } from './lines/LineGeometry.js';

function pcdFromArrays(xyz, colors, point_size, onLoad = undefined) {
    var pcd = new THREE.BufferGeometry();
    var xyz_ = [].concat.apply([], xyz);
    var colors_ = [].concat.apply([], colors).map(x => x / 255.0);
    pcd.setAttribute('position', new THREE.Float32BufferAttribute(xyz_, 3));
    pcd.setAttribute('color', new THREE.Float32BufferAttribute(colors_, 3));
    var material = new THREE.PointsMaterial({
        size: point_size,
        vertexColors: THREE.VertexColors,
        sizeAttenuation: true,
        alphaTest: 0.5,
        transparent: true
    });
    new THREE.TextureLoader().load('./disc.png', function (texture) {
        material.map = texture;
        material.needsUpdate = true;
        if (onLoad != undefined)
            onLoad();
    });
    return new THREE.Points(pcd, material);
}

function parsePose(tuple) {
    var q = tuple[0];
    var tvec = tuple[1];
    var T = new THREE.Matrix4();
    T.makeRotationFromQuaternion(new THREE.Quaternion(q[1], q[2], q[3], q[0]));
    T.setPosition(tvec[0], tvec[1], tvec[2]);
    return T.invert();
}

function setTransform(obj, T) {
    obj.position.setFromMatrixPosition(T);
    obj.quaternion.setFromRotationMatrix(T);
}

function drawCamera(T, corners, image_path, scale = 1., color = 0x000000, light = false) {
    var camera = new THREE.Group();
    camera.add(drawFrustum(corners, scale, color, light));
    if (image_path != undefined)
        camera.add(drawImagePlane(corners, image_path, scale));
    setTransform(camera, T);
    return camera;
}

function drawFrustum(corners, scale, color = 0x000000, light = false) {
    var c = new THREE.Vector3();
    var p3d = corners.map(x => new THREE.Vector3(x[0], x[1], 1.));
    p3d.push(c, p3d[3], p3d[0], c, p3d[2]);
    if (light) {
        var frustum = new THREE.Line(
            new THREE.BufferGeometry().setFromPoints(p3d),
            new THREE.LineBasicMaterial({color: color, linewidth: 10}));
    } else {
        const geometry = new LineGeometry();
        const positions = [];
        p3d.forEach(p => positions.push(p.x, p.y, p.z));
        geometry.setPositions(positions);
        frustum = new Line2(
            geometry, new LineMaterial({
                color: color,
                linewidth: 0.002,
                alphaToCoverage: true}));
    }
    frustum.geometry.scale(scale, scale, scale);
    return frustum;
}

function drawImagePlane(corners, image_path, scale) {
    var img = new THREE.MeshBasicMaterial({
        map: new THREE.TextureLoader().load(image_path),
        side: THREE.DoubleSide,
    });
    img.map.needsUpdate = true;
    var [w, h] = corners[2].map((x, i) => (x-corners[0][i])*scale);
    var plane = new THREE.Mesh(new THREE.PlaneGeometry(w, h), img);
    plane.overdraw = true;
    plane.rotateX(THREE.MathUtils.degToRad(180.));
    plane.translateZ(-scale);
    return plane;
}

function drawTrajectoryLine(T, T_prev, radius=0.02, color = 0x00ff00) {
    const start = new THREE.Vector3().setFromMatrixPosition(T);
    const end = new THREE.Vector3().setFromMatrixPosition(T_prev);
    const geometry = new LineGeometry();
    geometry.setPositions([start.x, start.y, start.z, end.x, end.y, end.z]);
    var line = new Line2(
        geometry,
        new LineMaterial({color: color, linewidth: 0.002, alphaToCoverage: true}));
    return line;
}

function drawRays(xyz, mask, position) {
    var skip = Math.max(Math.floor(xyz.length/40), 1);
    var ray_p3d = [];
    for (let i = 0; i < xyz.length; i += skip) {
        if ((mask[i] == 1) || (mask[i] === true)) {
            ray_p3d.push(new THREE.Vector3().fromArray(xyz[i]));
            ray_p3d.push(position);
        }
    }
    var rays = new THREE.LineSegments(
        new THREE.BufferGeometry().setFromPoints(ray_p3d),
        new THREE.LineBasicMaterial({color: 0xff0000, transparent: true, opacity: 0.5}));
    return rays;
}

export { parsePose, setTransform, drawCamera, drawTrajectoryLine, drawRays, pcdFromArrays};
