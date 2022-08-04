import './style.css'
import { quickStart, Vector3, Engine, Body } from 'surreal-engine';

const opts = {
  showFps: true,
  debug: {
    orbitControls: true
  }
};

const setupCameraAndLighting = (engine: Engine) => {
  engine.creator.directionalLight({
    color: '#ffffff',
    intensity: 1,
    pos: new Vector3(0, 30, 25),
    target: new Vector3(0, 0, 0),
    castShadow: true,
    shadowAreaHeight: 50,
    shadowAreaWidth: 40,
  })
  engine.creator.ambientLight({ color: '#ffffff', intensity: 0.5 });
  engine.setOrthographicCamera({
    distance: 12,
  });

  engine.setBackground({ color: '#000000' });
}

quickStart("#app", opts,
  assets => {
    assets.setBasePath('/assets/');
    assets.addTexture('asphalt', 'asphalt/ASPHALT_001_COLOR.jpg');
    assets.addTexture('asphalt@normal', 'asphalt/ASPHALT_001_NRM.jpg');
    assets.addTexture('asphalt@bump', 'asphalt/ASPHALT_001_DISP.png');
    assets.addTexture('asphalt@ao', 'asphalt/ASPHALT_001_OCC.jpg');
  },
  engine => {
    setupCameraAndLighting(engine);

    engine.materials.addTexturedMaterial('asphalt', {
      texture: {
        map: 'asphalt',
      },
      repeat: { x: 0, y: 0 },
    });

    engine.creator.box({
      pos: new Vector3(0, -1, 8),
      size: new Vector3(20, 0.2, 20),
      mass: 0,
      rigid: true,
      receiveShadow: true,
      material: 'asphalt',
    });


    const boxId = engine.creator.box({
      pos: new Vector3(0, 0, 0),
      size: new Vector3(1, 1, .3),
      mass: 1,
      rigid: true,
      castShadow: true,
      material: 'red',
    }).withOffsetCamera(new Vector3(-10, 10, -10))
      .withKeyboardMotion({ speed: 0.4, rotation: 0.5 })
      .id;


    const box = engine.manager.get(boxId)
    const dataset: any[] = [];
    engine.creator.timer(() => {
      const body = box.get(Body)
      const { _w: rotation } = body?.quaternion
      const { x, y, z } = body?.position
      dataset.push([x, y, z, rotation])
    }, 100, true)


    setTimeout(() => {
      console.log(dataset)
    }, 20000);


    // engine.manager.get
    // console.log(engine)

  }
);
