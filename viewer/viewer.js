import * as THREE from './jsm/three.module.js';
import { OrbitControls } from './jsm/OrbitControls.js';
import {parsePose, setTransform, drawCamera,
        drawTrajectoryLine, drawRays, pcdFromArrays} from './jsm/lib3d.js';


function drawPoints2d(canvas, image, p2ds, colors) {
    var ctx = canvas.getContext("2d");
    ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
    for (let i = 0; i < p2ds.length; i++) {
        ctx.fillStyle = colors[i];
        ctx.strokeStyle = colors[i];
        p2ds[i].forEach(function (p) {
            if ((p[0] >= 0) && (p[0] <= 1) && (p[1] >= 0) && (p[1] <= 1)) {
                ctx.beginPath();
                ctx.arc(p[0]*canvas.width, p[1]*canvas.height,
                        0.004*canvas.height, 0, 2 * Math.PI);
                ctx.fill();
            }
        });
    }
    ctx.stroke();
}

function getUrlParameter(sParam) {
    var sPageURL = window.location.search.substring(1),
        sURLVariables = sPageURL.split('&'),
        sParameterName,
        i;
    for (i = 0; i < sURLVariables.length; i++) {
        sParameterName = sURLVariables[i].split('=');
        if (sParameterName[0] === sParam) {
            return sParameterName[1] === undefined ? true : decodeURIComponent(sParameterName[1]);
        }
    }
}


class Viewer {
    camera
    controls
    scene
    group
    renderer
    dump
    dump_p2d
    nsteps

constructor(root_dir = './') {
    this.renderRequested = false;
    this.animation = {
        timer: undefined,
        index: 0,
        duration: 4,
    };
    this.data3d = {
        container: document.getElementById("3dCanvas"),
        pcd: undefined,
        point_size: 1,
        camera_scale: 1,
        T_query: undefined,
        frames: {
            init: undefined,
            query: undefined,
            refs: [],
        },
        trajectory: [],
        rays: undefined,
        loaded: false,
    };
    this.data2d = {
        images: {
            q: undefined,
            r: undefined,
        },
        canvas: {
            q: document.getElementById('qCanvas'),
            r: document.getElementById('rCanvas'),
        },
        loaded: false,
    };

    this.init3d();

    var data_dir = root_dir + 'sample/';
    var qname = getUrlParameter('query');
    if (qname != undefined) {
        console.log('Query: ', qname);
        data_dir = qname;
    }
    this.load(data_dir);
}

load(dir) {
    this.data_dir = dir;
    var path = this.data_dir + 'dump.json';
    console.log("Loading", path);
    var headers = {
        'Content-Type': 'application/json',
        'Accept-Encoding': 'gzip'
    };
    fetch(path, {headers: headers})
      .then(response => response.json())
      .then(json => this.onLoadDump(json));

    fetch(this.data_dir + 'dump_p2d.json', {headers: headers})
      .then(response => response.json())
      .then(json => this.onLoadPoints2d(json));
}

init3d() {
    var renderer = new THREE.WebGLRenderer( { antialias: true , preserveDrawingBuffer   : true} );
    renderer.setPixelRatio( window.devicePixelRatio );
    this.data3d.container.appendChild( renderer.domElement );

    var scene = new THREE.Scene();
    scene.background = new THREE.Color(0xFFFFFF);

    const axesHelper = new THREE.AxesHelper( 5 );
    scene.add(axesHelper);

    var camera = new THREE.PerspectiveCamera(
        45, this.data3d.container.clientWidth / this.data3d.container.clientHeight, .01, 1000);
    camera.position.set( 0, 0, -4 );
    camera.up.set(0, -1, 0);
    camera.lookAt(new THREE.Vector3(0, 0, 0));
    scene.add(camera);

    var controls = new OrbitControls(camera, renderer.domElement);
    controls.rotateSpeed = -0.5;
    controls.staticMoving = true;
    controls.autoRotate = false;
    controls.autoRotateSpeed = 4.0;

    this.renderer = renderer;
    this.scene = scene;
    this.camera = camera;
    this.controls = controls;

    this.onWindowResize();
    window.addEventListener('resize', this.onWindowResize.bind(this), false);
    controls.addEventListener('change', this.requestRender.bind(this));
    var container = renderer.domElement;
    container.setAttribute('tabindex', '-1');
    container.addEventListener('keypress', this.onKeyPress.bind(this));
}

clearScene() {
    if (this.group != undefined) {
        while(this.group.children.length > 0){
            this.group.remove(this.group.children[0]);
        }
        this.scene.remove(this.group);
    }
}

onLoadImage() {
    if (this.data2d.images.q.complete && this.data2d.images.r.complete) {
        console.log('Images loaded');
        this.data2d.loaded = true;
        this.onWindowResize();
        this.onWindowResize();  // second call to fix flex-renderer issues
    }
}

onLoadDump(dump) {
    console.log('Dump loaded');
    this.dump = dump;
    this.clearScene();
    this.group = new THREE.Group();
    this.scene.add(this.group);

    var loaded_count = 0;
    this.data2d.images.q = new Image();
    this.data2d.images.r = new Image();
    this.data2d.images.q.onload = this.onLoadImage.bind(this);
    this.data2d.images.r.onload = this.onLoadImage.bind(this);
    this.data2d.images.q.src = this.data_dir + dump.image.query;
    this.data2d.images.r.src = this.data_dir + dump.image.refs[0];

    this.nsteps = dump.T.query.length;

    this.data3d.T_query = dump.T.query.map(tuple => parsePose(tuple));
    var T_final = this.data3d.T_query[this.nsteps-1];
    this.camera.up.set(0, 1, 0);
    this.camera.position.copy(new THREE.Vector3(0, -10, -20).applyMatrix4(T_final));
    this.controls.target.setFromMatrixPosition(T_final);

    this.data3d.frames.refs = [];
    dump.T.refs.forEach(function (tuple, i) {
        var T = parsePose(tuple);
        let camera = drawCamera(
            T, dump.camera.refs[i], this.data_dir + dump.image.refs[i], this.data3d.camera_scale);
        this.data3d.frames.refs.push(camera);
        this.group.add(camera);
    }, this);

    this.data3d.frames.init = drawCamera(
        this.data3d.T_query[0], dump.camera.query, undefined, this.data3d.camera_scale, 0x0000ff);
    this.data3d.frames.query = drawCamera(
        T_final, dump.camera.query, this.data_dir + dump.image.query, this.data3d.camera_scale, 0xff0000);
    this.group.add(this.data3d.frames.init);
    this.group.add(this.data3d.frames.query);

    this.data3d.trajectory = [];
    this.data3d.T_query.forEach(function (T, i) {
        if (i > 0) {
            var line = drawTrajectoryLine(T, this.data3d.T_query[i-1]);
            this.group.add(line);
            this.data3d.trajectory.push(line);
        }
    }, this);

    this.data3d.pcd = pcdFromArrays(
        dump.p3d.xyz, dump.p3d.colors, this.data3d.point_size,
        this.requestRender.bind(this));
    this.group.add(this.data3d.pcd);

    this.data3d.loaded = true;
    this.requestRender();
    console.log('Done display');
}

onLoadPoints2d(dump) {
    this.dump_p2d = dump;
}

render() {
    this.renderRequested = false;
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
}

requestRender() {
    if (!this.renderRequested) {
        this.renderRequested = true;
        requestAnimationFrame(this.render.bind(this));
    }
}

init2d() {
    drawPoints2d(this.data2d.canvas.q, this.data2d.images.q, [this.dump.p2d.query], ["red"]);
    drawPoints2d(this.data2d.canvas.r, this.data2d.images.r, [this.dump.p2d.refs[0]], ["lime"]);
}

clear2d() {
    var canvas = this.data2d.canvas.q;
    var image = this.data2d.images.q;
    canvas.getContext("2d").drawImage(image, 0, 0, canvas.width, canvas.height);
}


onWindowResize() {
    if (this.data2d.loaded) {
        var pane2d = document.getElementById("pane2d");
        var parentStyle = window.getComputedStyle(pane2d.parentElement);
        var [W, H] = [parseFloat(parentStyle.width), parseFloat(parentStyle.height)];
        var pad = 5;
        var max_width = parseFloat(window.getComputedStyle(pane2d).maxWidth);
        var [H_max, W_max] = [H/2-pad*2, W/100*max_width-pad*2];
        var aspect_max = W_max / H_max;
        var aspect1 = this.data2d.images.q.width / this.data2d.images.q.height;
        var aspect2 = this.data2d.images.r.width / this.data2d.images.r.height;
        if (aspect1 > aspect_max) {
            var [w1, h1] = [W_max, W_max/aspect1];
        } else {
            var [w1, h1] = [aspect1*H_max, H_max];
        }
        if (aspect2 > aspect_max) {
            var [w2, h2] = [W_max, W_max/aspect2];
        } else {
            var [w2, h2] = [aspect2*H_max, H_max];
        }
        pane2d.style.width = Math.floor(Math.max(w1, w2)+pad*2).toString()+"px";
        this.data2d.canvas.q.width = w1;
        this.data2d.canvas.q.height = h1;
        this.data2d.canvas.r.width = w2;
        this.data2d.canvas.r.height = h2;
        this.init2d();
    }

    var w = this.data3d.container.clientWidth;
    var h = this.data3d.container.clientHeight;
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h);
    this.requestRender();
}

onKeyPress(ev) {
    switch ( ev.key || String.fromCharCode( ev.keyCode || ev.charCode ) ) {
        case '-':
            this.updatePointSize(-0.1);
            break;
        case '+':
        case '=':
            this.updatePointSize(0.1);
            break;
        case '[':
            this.scaleCameras(0.5);
            break;
        case ']':
            this.scaleCameras(2.);
            break;
        case 'i':
            this.animation.index = 0;
        case 'o':
            this.animStep();
            break;
        case 'r':
            this.controls.autoRotate = !this.controls.autoRotate;
            this.requestRender();
            break;
        case 'p':
            if (this.data3d.loaded && this.data2d.loaded) {
                if (this.animation.timer === undefined) {
                    this.animStart();
                } else {
                    this.animClear();
                }
            }
            break;
        case 'h':
            var elem = document.getElementById("info")
            elem.style.display = (elem.style.display === 'none') ? 'block' : 'none';
            break;
        case 'a':
            if (this.data3d.frames.query.visible) {
                this.data3d.trajectory.forEach(line => {line.visible = false});
                this.clear2d();
            } else {
                this.data3d.trajectory.forEach(line => {line.visible = true});
                this.init2d();
            }
            this.data3d.frames.query.visible = !this.data3d.frames.query.visible;
            this.data3d.frames.init.visible = !this.data3d.frames.init.visible;
            this.requestRender();
            break;
    }
}

updatePointSize(d) {
    this.data3d.point_size = Math.max(this.data3d.point_size + d, 0.1);
    if (this.data3d.pcd)
        this.data3d.pcd.material.size = this.data3d.point_size;
    this.requestRender();
}

scaleCameras(s) {
    this.data3d.camera_scale = Math.max(this.data3d.camera_scale*s, 0.0001);
    if (this.data3d.frames.query)
        this.data3d.frames.query.scale.multiplyScalar(s);
    if (this.data3d.frames.init)
        this.data3d.frames.init.scale.multiplyScalar(s);
    this.data3d.frames.refs.forEach(frame => frame.scale.multiplyScalar(s));
    this.requestRender();
}

animStart() {
    this.animation.index = 0;
    this.animation.timer = setInterval(this.animStep.bind(this), (1000*this.animation.duration)/this.nsteps);
}

animStep() {
    let index = this.animation.index;
    if (index < this.nsteps) {
        this.group.remove(this.data3d.rays);
        if (index === 0) {
            this.data3d.trajectory.forEach(line => {line.visible = false});
        } else {
            this.data3d.trajectory[index-1].visible = true;
        }
        setTransform(this.data3d.frames.query, this.data3d.T_query[index]);
        drawPoints2d(
            this.data2d.canvas.q, this.data2d.images.q,
            [this.dump_p2d.query[this.nsteps-1], this.dump_p2d.query[index]],
            ["lime", "red"]);
        if ('masks' in this.dump_p2d) {
            this.data3d.rays = drawRays(
                this.dump.p3d.xyz, this.dump_p2d.masks[index], this.data3d.frames.query.position);
            this.group.add(this.data3d.rays);
        }
        this.requestRender();
    } else if (index > Math.floor(1.1*this.nsteps)) {
        this.animation.index = -1;
        this.group.remove(this.data3d.rays);
    }
    this.animation.index++;
}

animClear() {
    if (this.animation.timer != undefined) {
        clearInterval(this.animation.timer);
        this.animation.timer = undefined;
    }
    this.data3d.trajectory.forEach(line => {line.visible = true});
    setTransform(this.data3d.frames.query, this.data3d.T_query[this.nsteps-1]);
    this.group.remove(this.data3d.rays);
    this.requestRender();
    this.init2d();
}
}

//window.Viewer = Viewer;
export {Viewer};
