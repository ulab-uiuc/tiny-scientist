import React, { useRef, useState, useMemo, useEffect, useCallback } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { TrackballControls, Text, Line, OrthographicCamera, Billboard } from '@react-three/drei';
import * as THREE from 'three';
import userActionTracker from '../utils/userActionTracker';

// --- Constants ---
const CUBE_SIZE = 100;
const HALF_SIZE = CUBE_SIZE / 2;
const NODE_RANGE = 90;
const LABEL_COLOR = '#374151';
const GRID_COLOR = '#e5e7eb';
const CROSSHAIR_COLOR = '#9ca3af';

// Keep node coloring consistent with Exploration view
const getNodeColor = (node, colorMap) => {
  if (!node) return '#FF6B6B';
  if (node.isMergedResult) return '#B22222';
  if (node.isNewlyGenerated || node.isModified) return '#FFD700';
  return (colorMap && colorMap[node.type]) || '#FF6B6B';
};

// --- Helpers ---
const scoreToPos = (score) => {
  // Score range: -50 to 50 → Position range: -NODE_RANGE/2 to NODE_RANGE/2
  const clamped = Math.max(-50, Math.min(50, score));
  return (clamped / 50) * (NODE_RANGE / 2);
};
const posToScore = (pos) => Math.max(-50, Math.min(50, (pos / (NODE_RANGE / 2)) * 50));
const isFiniteNumber = (value) => typeof value === 'number' && Number.isFinite(value);
const hasVector2 = (pos) => !!pos && isFiniteNumber(pos.x) && isFiniteNumber(pos.y);
const hasVector3 = (pos) => !!pos && isFiniteNumber(pos.x) && isFiniteNumber(pos.y) && isFiniteNumber(pos.z);

// --- Components ---

const CameraController = ({ targetFaceIndex, targetUp, isSnapping, setSnapped, setIsSnapping, controlsRef }) => {
  const { camera } = useThree();

  // 6 orthogonal view positions
  const faceConfigs = useMemo(() => [
    { pos: [0, 0, 1] },   // Front
    { pos: [1, 0, 0] },   // Right
    { pos: [0, 0, -1] },  // Back
    { pos: [-1, 0, 0] },  // Left
    { pos: [0, 1, 0] },   // Top
    { pos: [0, -1, 0] },  // Bottom
  ], []);

  useFrame(() => {
    if (isSnapping && targetFaceIndex !== null && targetUp) {
      const R = 200;
      const config = faceConfigs[targetFaceIndex];
      const targetPos = new THREE.Vector3(...config.pos).multiplyScalar(R);

      // Lerp position and up vector
      camera.position.lerp(targetPos, 0.15);
      camera.up.lerp(targetUp, 0.15);

      // Sync OrbitControls internal state
      if (controlsRef?.current) {
        controlsRef.current.update();
      }

      const dist = camera.position.distanceTo(targetPos);

      if (dist < 0.5) {
        camera.position.copy(targetPos);
        camera.up.copy(targetUp);
        if (controlsRef?.current) {
          controlsRef.current.update();
        }
        setSnapped(true);
        setIsSnapping(false);
      }
    }
  });
  return null;
};

const WireframeBox = () => {
  return (
    <lineSegments>
      <edgesGeometry args={[new THREE.BoxGeometry(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE)]} />
      <lineBasicMaterial color={GRID_COLOR} />
    </lineSegments>
  );
};

const ProjectedAxis = ({ type }) => {
  const H = HALF_SIZE;
  const points = useMemo(() => {
    if (type === 'x') return [[-H, 0, 0], [H, 0, 0]];
    if (type === 'y') return [[0, -H, 0], [0, H, 0]];
    if (type === 'z') return [[0, 0, -H], [0, 0, H]];
    return [[0, 0, 0], [0, 0, 0]];
  }, [type]);

  return (
    <group>
      <Line points={points} color={CROSSHAIR_COLOR} lineWidth={1.5} />
    </group>
  );
};

const AxisLabel = ({ position, text, rotationZ, anchorX, anchorY, fontSize = 3.5 }) => {
  const { camera } = useThree();
  const ref = useRef();
  const rotQuat = useMemo(() => new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 0, 1), rotationZ), [rotationZ]);

  useFrame(() => {
    if (!ref.current) return;
    ref.current.quaternion.copy(camera.quaternion);
    if (rotationZ !== 0) {
      ref.current.quaternion.multiply(rotQuat);
    }
  });

  return (
    <group ref={ref} position={position}>
      <Text fontSize={fontSize} color={LABEL_COLOR} anchorX={anchorX} anchorY={anchorY} fontWeight="bold" outlineWidth={0.25} outlineColor="#ffffff">
        {text}
      </Text>
    </group>
  );
};

const AxisLabels3D = ({ activeFace, dims, isSnapped }) => {
  const faceNormals = [
    { id: 0, v: new THREE.Vector3(0, 0, 1), perpAxis: 'z' },   // Front
    { id: 1, v: new THREE.Vector3(1, 0, 0), perpAxis: 'x' },   // Right
    { id: 2, v: new THREE.Vector3(0, 0, -1), perpAxis: 'z' },  // Back
    { id: 3, v: new THREE.Vector3(-1, 0, 0), perpAxis: 'x' },  // Left
    { id: 4, v: new THREE.Vector3(0, 1, 0), perpAxis: 'y' },   // Top
    { id: 5, v: new THREE.Vector3(0, -1, 0), perpAxis: 'y' },  // Bottom
  ];

  const axisDefs = [
    { key: 'x', vec: new THREE.Vector3(1, 0, 0), dimIndex: 0 },
    { key: 'y', vec: new THREE.Vector3(0, 1, 0), dimIndex: 1 },
    { key: 'z', vec: new THREE.Vector3(0, 0, 1), dimIndex: 2 }
  ];

  const faceNormal = faceNormals[activeFace].v || faceNormals[0].v;

  return (
    <>
      {axisDefs.map(({ key, vec, dimIndex }) => {
        const dim = dims[dimIndex];
        if (!dim) return null;

        const isPerpendicular = Math.abs(vec.clone().normalize().dot(faceNormal)) > 0.999;
        if (isSnapped && isPerpendicular) return null;

        const posA = vec.clone().multiplyScalar(-HALF_SIZE);
        const posB = vec.clone().multiplyScalar(HALF_SIZE);

        return (
          <React.Fragment key={key}>
            <AxisLabel position={posA} text={dim.dimensionA} rotationZ={0} anchorX="center" anchorY="middle" fontSize={3} />
            <AxisLabel position={posB} text={dim.dimensionB} rotationZ={0} anchorX="center" anchorY="middle" fontSize={3} />
          </React.Fragment>
        );
      })}
    </>
  );
};

const Node3D = ({ node, position, color, isSelected, isHovered, onPointerOver, onPointerOut, onClick, onDragEnd, axisMapping, isSnapped, allNodes, dims, setIsDraggingNode, setDragVisual3D, setDragVisualState, projectToFace, dragVisual3D, dragVisualState, activeCount, activeDimensionIndices, dragHoverTarget }) => {
  const meshRef = useRef();
  const [isDragging, setIsDragging] = useState(false);
  const { camera, raycaster } = useThree();
  const dragPlane = useMemo(() => new THREE.Plane(), []);
  const intersection = useMemo(() => new THREE.Vector3(), []);
  const dragOffset = useMemo(() => new THREE.Vector3(), []);
  const dragStart = useRef(new THREE.Vector3());
  const movedRef = useRef(false);

  // Nodes stay at their true 3D position (no projection to face)
  const targetPos = new THREE.Vector3(...position);

  useFrame(() => {
    if (meshRef.current && !isDragging) {
      meshRef.current.position.lerp(targetPos, 0.15);
    }
  });

  // Get normal vector for the perpendicular axis
  const getPerpendicularNormal = (axis) => {
    switch (axis) {
      case 'x': return new THREE.Vector3(1, 0, 0);
      case 'y': return new THREE.Vector3(0, 1, 0);
      case 'z': return new THREE.Vector3(0, 0, 1);
      default: return new THREE.Vector3(0, 0, 1);
    }
  };

  const handlePointerDown = (e) => {
    e.stopPropagation();
    if (e.button !== 0) return;

    // Prevent canvas mousedown from triggering rotation
    e.nativeEvent?.stopImmediatePropagation();

    setIsDragging(true);
    if (setIsDraggingNode) setIsDraggingNode(true);
    movedRef.current = false;

    let normal;
    if (activeCount === 1) {
      normal = new THREE.Vector3(0, 0, 1); // Force drag plane to be XY for 1D (allowing X movement)
    } else {
      normal = isSnapped && axisMapping
        ? getPerpendicularNormal(axisMapping.perpAxis)
        : new THREE.Vector3().copy(camera.position).normalize();
    }
    dragPlane.setFromNormalAndCoplanarPoint(normal, meshRef.current.position);
    if (raycaster.ray.intersectPlane(dragPlane, intersection)) {
      dragOffset.copy(intersection).sub(meshRef.current.position);
    }
    dragStart.current.copy(meshRef.current.position);
    e.target.setPointerCapture(e.pointerId);
  };

  const handlePointerMove = (e) => {
    if (!isDragging) return;
    e.stopPropagation();
    if (raycaster.ray.intersectPlane(dragPlane, intersection)) {
      const newPos = intersection.sub(dragOffset);
      const lim = NODE_RANGE / 2;
      newPos.x = Math.max(-lim, Math.min(lim, newPos.x));

      // For 1D view, restrict dragging to X-axis only
      if (activeCount === 1) {
        newPos.y = 0; // Force Y to 0
        newPos.z = 0; // Force Z to 0
      } else {
        newPos.y = Math.max(-lim, Math.min(lim, newPos.y));
        newPos.z = Math.max(-lim, Math.min(lim, newPos.z));
      }
      meshRef.current.position.copy(newPos);
      movedRef.current = true;

      // live visual for modify/merge - lock perpendicular axis
      const curr = newPos.clone();
      if (isSnapped && axisMapping) {
        if (axisMapping.perpAxis === 'x') curr.x = position[0];
        if (axisMapping.perpAxis === 'y') curr.y = position[1];
        if (axisMapping.perpAxis === 'z') curr.z = position[2];
      }
      let mergeTargetId = null;
      let mergeTargetPos = null;
      let best = Infinity;
      const thresh = 8;
      allNodes.forEach(other => {
        if (other.id === node.id || other.isGhost) return;
        const px = scoreToPos(other.scores?.[`${dims[0]?.dimensionA}-${dims[0]?.dimensionB}`] ?? 0);
        const py = scoreToPos(other.scores?.[`${dims[1]?.dimensionA}-${dims[1]?.dimensionB}`] ?? 0);
        const pz = scoreToPos(other.scores?.[`${dims[2]?.dimensionA}-${dims[2]?.dimensionB}`] ?? 0);
        const facePos = projectToFace(new THREE.Vector3(px, py, pz));
        const dragFace = projectToFace(curr);
        const d = Math.hypot(dragFace.a - facePos.a, dragFace.b - facePos.b);
        if (d < thresh && d < best) {
          best = d;
          mergeTargetId = other.id;
          mergeTargetPos = new THREE.Vector3(px, py, pz);
        }
      });
      if (setDragVisual3D) {
        if (mergeTargetId) {
          setDragVisual3D({ type: 'merge', sourceId: node.id, current: curr.clone(), targetId: mergeTargetId, targetPos: mergeTargetPos });
        } else {
          setDragVisual3D({ type: 'modify', sourceId: node.id, start: dragStart.current.clone(), current: curr.clone() });
        }
      }
      if (setDragVisualState) {
        if (mergeTargetId && mergeTargetPos) {
          setDragVisualState({
            type: 'merge',
            sourceNodeId: node.id,
            targetNodeId: mergeTargetId,
            ghostPosition: { x: dragStart.current.x, y: dragStart.current.y, z: dragStart.current.z },
            targetPosition: { x: mergeTargetPos.x, y: mergeTargetPos.y, z: mergeTargetPos.z }
          });
        } else {
          setDragVisualState({
            type: 'modify',
            sourceNodeId: node.id,
            ghostPosition: { x: dragStart.current.x, y: dragStart.current.y, z: dragStart.current.z },
            newPosition: { x: curr.x, y: curr.y, z: curr.z }
          });
        }
      }
    }
  };

  const handlePointerUp = (e) => {
    if (!isDragging) return;
    e.stopPropagation();
    setIsDragging(false);
    if (setIsDraggingNode) setIsDraggingNode(false);
    e.target.releasePointerCapture(e.pointerId);
    if (!movedRef.current) {
      if (onClick) onClick(node);
      return;
    }
    const curr = meshRef.current.position.clone();

    // Lock perpendicular axis when snapped
    if (isSnapped && activeCount > 1 && axisMapping) { // Apply for 2D/3D only
      if (axisMapping.perpAxis === 'x') curr.x = position[0];
      if (axisMapping.perpAxis === 'y') curr.y = position[1];
      if (axisMapping.perpAxis === 'z') curr.z = position[2];
    } else if (activeCount === 1) { // For 1D, ensure Y and Z are 0
      curr.y = 0;
      curr.z = 0;
    }

    // Build new score map
    const scoreFromAxis = (val) => posToScore(val);
    const scoresMap = {};
    if (activeCount === 1) {
      const activeDimIndex = activeDimensionIndices[0];
      if (dims[activeDimIndex]) {
        scoresMap[`${dims[activeDimIndex].dimensionA}-${dims[activeDimIndex].dimensionB}`] = scoreFromAxis(curr.x);
      }
    } else { // 2D or 3D
      if (dims[0]) scoresMap[`${dims[0].dimensionA}-${dims[0].dimensionB}`] = scoreFromAxis(curr.x);
      if (dims[1]) scoresMap[`${dims[1].dimensionA}-${dims[1].dimensionB}`] = scoreFromAxis(curr.y);
      if (dims[2]) scoresMap[`${dims[2].dimensionA}-${dims[2].dimensionB}`] = scoreFromAxis(curr.z);
    }

    // Merge detection in 2D plane (use passed-in projectToFace)
    const dragged2D = projectToFace(curr);
    let mergeTargetId = null;
    let mergeTargetPos = null;
    let bestDist = Infinity;
    const threshold = 8; // scene units

    allNodes.forEach(other => {
      if (other.id === node.id || other.isGhost) return;
      const targetVec = new THREE.Vector3(
        scoreToPos(other.scores?.[`${dims[0]?.dimensionA}-${dims[0]?.dimensionB}`] ?? 0),
        scoreToPos(other.scores?.[`${dims[1]?.dimensionA}-${dims[1]?.dimensionB}`] ?? 0),
        scoreToPos(other.scores?.[`${dims[2]?.dimensionA}-${dims[2]?.dimensionB}`] ?? 0)
      );
      const pos = projectToFace(targetVec);
      const dx = dragged2D.a - pos.a;
      const dy = dragged2D.b - pos.b;
      const d = Math.hypot(dx, dy);
      if (d < threshold && d < bestDist) {
        bestDist = d;
        mergeTargetId = other.id;
        mergeTargetPos = targetVec;
      }
    });

    if (setDragVisual3D) {
      if (mergeTargetId && mergeTargetPos) {
        setDragVisual3D({ type: 'merge', sourceId: node.id, current: curr.clone(), targetId: mergeTargetId, targetPos: mergeTargetPos.clone(), hold: true });
      } else {
        setDragVisual3D({ type: 'modify', sourceId: node.id, start: dragStart.current.clone(), current: curr.clone(), hold: true });
      }
    }
    if (onDragEnd) onDragEnd(node.id, { scoresMap, mergeTargetId, clientX: e.clientX, clientY: e.clientY }, e);
  };

  return (
    <group>
      <mesh ref={meshRef} position={position} userData={{ nodeId: node.id }}
        onPointerDown={handlePointerDown} onPointerMove={handlePointerMove} onPointerUp={handlePointerUp}
        onPointerOver={(e) => { e.stopPropagation(); onPointerOver(node) }}
        onPointerOut={(e) => { e.stopPropagation(); onPointerOut() }}
        scale={
          (dragVisualState?.type === 'merge' && dragVisualState.targetNodeId === node.id) ||
            (dragVisual3D?.type === 'merge' && dragVisual3D.targetId === node.id) ||
            (dragHoverTarget === node.id)
            ? 1.35
            : ((isHovered || isDragging) && !isSelected ? (activeCount === 1 ? 1.5 : 1.2) : (activeCount === 1 ? 1.2 : 1))
        }
      >
        <sphereGeometry args={[isSelected ? 5 : (activeCount === 1 ? 4 : 3.5), 32, 32]} />
        <meshBasicMaterial
          color={color}
          opacity={
            (dragVisualState?.type === 'merge' && dragVisualState.sourceNodeId === node.id) ||
              (dragVisual3D?.type === 'merge' && dragVisual3D.sourceId === node.id)
              ? 0.15
              : 1
          }
          transparent={
            (dragVisualState?.type === 'merge' && dragVisualState.sourceNodeId === node.id) ||
            (dragVisual3D?.type === 'merge' && dragVisual3D.sourceId === node.id)
          }
          toneMapped={false}
        />
        {isSelected && (
          <mesh scale={[1.08, 1.08, 1.08]}>
            <sphereGeometry args={[isSelected ? 5.3 : (activeCount === 1 ? 4.3 : 3.8), 32, 32]} />
            <meshBasicMaterial color="#000" side={THREE.BackSide} />
          </mesh>
        )}
      </mesh>
      <Billboard position={position}>
        <Text
          fontSize={3}
          color="#1f2937"
          fillOpacity={isHovered || isSelected ? 1 : 0.3}
          outlineWidth={0.2}
          outlineColor="#ffffff"
          anchorY="bottom"
          position={[0, (activeCount === 1 ? 4 : 5), 0]} // Adjusted position for 1D
        >
          {node.title}
        </Text>
      </Billboard>
    </group>
  );
};

const SceneContent = ({
  nodes,
  dimensions,
  onNodeDragEnd,
  selectedNode,
  onNodeClick,
  hoveredNode,
  onNodeHover,
  targetFaceIndex,
  setTargetFaceIndex,
  pendingChange,
  pendingMerge,
  colorMap,
  dragVisualState,
  setDragVisualState,
  mergeAnimationState,
  activeDimensionIndices,
  onDropExternal,
  dragHoverTarget,
  onDragHover
}) => {
  const { camera, gl, scene, raycaster } = useThree();
  const controlsRef = useRef();
  const [isSnapped, setSnapped] = useState(true);
  const [isSnapping, setIsSnapping] = useState(false);
  const [isDraggingNode, setIsDraggingNode] = useState(false);
  const [dragVisual3D, setDragVisual3D] = useState(null);
  const [targetUp, setTargetUp] = useState(null);
  const [axisMapping, setAxisMapping] = useState({
    perpAxis: 'z',
    horizontalAxis: 'x',
    verticalAxis: 'y'
  });

  // Handle external drops (HTML5 DnD)
  useEffect(() => {
    const canvas = gl.domElement;
    // Prevent global DOM tracker from double-counting canvas-level events; track nodes semantically instead.
    canvas.setAttribute('data-tracking-name', 'evaluation_3d_canvas');
    canvas.setAttribute('data-uatrack-suppress-click', 'true');
    canvas.setAttribute('data-uatrack-suppress-hover', 'true');
    canvas.setAttribute('data-uatrack-suppress-drag', 'true');

    const handleDragOver = (e) => {
      e.preventDefault();
      e.stopPropagation();

      if (!onDragHover) return;

      const rect = canvas.getBoundingClientRect();
      const mouse = new THREE.Vector2(
        ((e.clientX - rect.left) / rect.width) * 2 - 1,
        -((e.clientY - rect.top) / rect.height) * 2 + 1
      );

      raycaster.setFromCamera(mouse, camera);

      const intersects = raycaster.intersectObjects(scene.children, true);
      const hit = intersects.find(intersect =>
        intersect.object.userData && intersect.object.userData.nodeId
      );

      if (hit) {
        onDragHover(hit.object.userData.nodeId);
      } else {
        onDragHover(null);
      }
    };

    const handleDrop = (e) => {
      e.preventDefault();
      e.stopPropagation();

      // Clear hover state on drop
      if (onDragHover) onDragHover(null);

      if (!onDropExternal) return;

      const rect = canvas.getBoundingClientRect();
      const mouse = new THREE.Vector2(
        ((e.clientX - rect.left) / rect.width) * 2 - 1,
        -((e.clientY - rect.top) / rect.height) * 2 + 1
      );

      raycaster.setFromCamera(mouse, camera);

      // Filter for objects that are likely our nodes
      const intersects = raycaster.intersectObjects(scene.children, true);

      // Find the first intersected object that has nodeId in userData
      const hit = intersects.find(intersect =>
        intersect.object.userData && intersect.object.userData.nodeId
      );

      if (hit) {
        const nodeId = hit.object.userData.nodeId;
        const targetNode = nodes.find(n => n.id === nodeId);
        if (targetNode) {
          onDropExternal(e, targetNode);
          return;
        }
      }

      // If no valid node hit, drop on empty space
      onDropExternal(e, null);
    };

    canvas.addEventListener('dragover', handleDragOver);
    canvas.addEventListener('drop', handleDrop);
    // Also handle dragleave to clear hover
    const handleDragLeave = () => {
      if (onDragHover) onDragHover(null);
    };
    canvas.addEventListener('dragleave', handleDragLeave);

    return () => {
      canvas.removeEventListener('dragover', handleDragOver);
      canvas.removeEventListener('drop', handleDrop);
      canvas.removeEventListener('dragleave', handleDragLeave);
    };
  }, [gl, camera, scene, raycaster, nodes, onDropExternal, onDragHover]);

  // Compute axis mapping based on camera orientation
  const computeAxisMapping = useCallback((cam) => {
    const direction = new THREE.Vector3();
    cam.getWorldDirection(direction);

    const up = cam.up.clone().normalize();
    const right = new THREE.Vector3().crossVectors(direction, up).normalize();

    const axes = ['x', 'y', 'z'];
    const vecs = {
      x: new THREE.Vector3(1, 0, 0),
      y: new THREE.Vector3(0, 1, 0),
      z: new THREE.Vector3(0, 0, 1)
    };

    // 1. perpAxis: axis most aligned with view direction
    let perpAxis = 'z', maxDirDot = 0;
    axes.forEach(axis => {
      const dot = Math.abs(direction.dot(vecs[axis]));
      if (dot > maxDirDot) {
        maxDirDot = dot;
        perpAxis = axis;
      }
    });

    // 2. horizontalAxis: remaining axis most aligned with screen right
    const remaining = axes.filter(a => a !== perpAxis);
    let horizontalAxis = remaining[0], maxRightDot = 0;
    remaining.forEach(axis => {
      const dot = Math.abs(right.dot(vecs[axis]));
      if (dot > maxRightDot) {
        maxRightDot = dot;
        horizontalAxis = axis;
      }
    });

    // 3. verticalAxis: the remaining one
    const verticalAxis = remaining.find(a => a !== horizontalAxis);

    return { perpAxis, horizontalAxis, verticalAxis };
  }, []);

  // Compute target up vector based on current camera orientation
  const computeTargetUp = useCallback((cam, perpAxis) => {
    // Get actual screen Y direction from camera matrix (not camera.up)
    const screenUp = new THREE.Vector3();
    cam.matrixWorld.extractBasis(new THREE.Vector3(), screenUp, new THREE.Vector3());
    screenUp.normalize();

    // Two axes on screen (non-perp)
    const axes = ['x', 'y', 'z'].filter(a => a !== perpAxis);
    const vecs = {
      x: new THREE.Vector3(1, 0, 0),
      y: new THREE.Vector3(0, 1, 0),
      z: new THREE.Vector3(0, 0, 1)
    };

    // Find which axis is closer to screen vertical direction
    const dot0 = Math.abs(screenUp.dot(vecs[axes[0]]));
    const dot1 = Math.abs(screenUp.dot(vecs[axes[1]]));
    const verticalAxis = dot0 > dot1 ? axes[0] : axes[1];

    // Determine sign based on dot product (preserve visual direction)
    const dot = screenUp.dot(vecs[verticalAxis]);
    return dot > 0 ? vecs[verticalAxis].clone() : vecs[verticalAxis].clone().negate();
  }, []);

  const handleEnd = () => {
    const camDir = camera.position.clone().normalize();
    const directions = [
      { id: 0, v: new THREE.Vector3(0, 0, 1), perpAxis: 'z' },   // Front
      { id: 1, v: new THREE.Vector3(1, 0, 0), perpAxis: 'x' },   // Right
      { id: 2, v: new THREE.Vector3(0, 0, -1), perpAxis: 'z' },  // Back
      { id: 3, v: new THREE.Vector3(-1, 0, 0), perpAxis: 'x' },  // Left
      { id: 4, v: new THREE.Vector3(0, 1, 0), perpAxis: 'y' },   // Top
      { id: 5, v: new THREE.Vector3(0, -1, 0), perpAxis: 'y' },  // Bottom
    ];

    let best = 0, maxDot = -Infinity;
    directions.forEach(d => {
      const dot = camDir.dot(d.v);
      if (dot > maxDot) {
        maxDot = dot;
        best = d.id;
      }
    });

    // Compute axis mapping based on current camera orientation
    const mapping = computeAxisMapping(camera);
    setAxisMapping(mapping);

    // Only snap if very close to an orthogonal face (threshold: 0.95)
    const SNAP_THRESHOLD = 0.95;
    const shouldSnap = maxDot > SNAP_THRESHOLD;

    if (shouldSnap) {
      // Compute target up vector based on current screen orientation
      const up = computeTargetUp(camera, directions[best].perpAxis);
      setTargetUp(up);

      setTargetFaceIndex(best);
      setIsSnapping(true);
      setSnapped(false);
    }

    // Helper to get dimension info
    const axisToIndex = { x: 0, y: 1, z: 2 };
    const getDimInfo = (axisName) => {
      const index = axisToIndex[axisName];
      const dim = dimensions[index];
      if (!dim) return null;
      return {
        id: dim.id, // Assuming dimensions might have IDs, otherwise optional
        label: `${dim.dimensionA} vs ${dim.dimensionB}`,
        dimensionA: dim.dimensionA,
        dimensionB: dim.dimensionB
      };
    };

    userActionTracker.trackAction('3d_rotate', 'evaluation_3d_view', {
      axisMapping: mapping,
      targetFaceIndex: best,
      snapped: shouldSnap,
      snapDistance: maxDot,
      viewDimensions: {
        horizontal: getDimInfo(mapping.horizontalAxis),
        vertical: getDimInfo(mapping.verticalAxis),
        depth: getDimInfo(mapping.perpAxis)
      }
    });
  };

  const handleStart = () => {
    setSnapped(false);
    setIsSnapping(false);
  };

  useEffect(() => {
    const canvas = document.querySelector('canvas');
    if (!canvas) return;

    const handleWheel = (e) => {
      e.preventDefault();
      camera.zoom = Math.max(1, Math.min(10, camera.zoom - e.deltaY * 0.005));
      camera.updateProjectionMatrix();
    };

    canvas.addEventListener('wheel', handleWheel, { passive: false });

    return () => {
      canvas.removeEventListener('wheel', handleWheel);
    };
  }, [camera]);

  // Project 3D position to 2D plane based on axis mapping
  const projectToPlane = useCallback((vec) => {
    const getVal = (axis) => axis === 'x' ? vec.x : axis === 'y' ? vec.y : vec.z;
    return {
      a: getVal(axisMapping.horizontalAxis),
      b: getVal(axisMapping.verticalAxis)
    };
  }, [axisMapping]);

  // Filter out root node once at the top level
  const validNodes = useMemo(() => nodes.filter(node => node.type !== 'root'), [nodes]);

  // Clear internal dragVisual3D when parent clears dragVisualState (on cancel/error)
  useEffect(() => {
    if (!dragVisualState && dragVisual3D) {
      setDragVisual3D(null);
    }
  }, [dragVisualState]); // Remove dragVisual3D from dependencies to avoid unnecessary reruns

  // Camera Transition Logic based on Active Dimensions
  const activeCount = activeDimensionIndices ? activeDimensionIndices.length : 3;

  const mergeTargetPosition = useMemo(() => {
    if (!dragVisualState || dragVisualState.type !== 'merge') return null;
    if (hasVector3(dragVisualState.targetPosition)) return dragVisualState.targetPosition;
    if (!dragVisualState.targetNodeId) return null;
    const targetNode = nodes.find(n => n.id === dragVisualState.targetNodeId);
    if (!targetNode) return null;

    const getScore = (dimObj) => {
      if (!dimObj) return 0;
      const key = `${dimObj.dimensionA}-${dimObj.dimensionB}`;
      if (targetNode.scores && targetNode.scores[key] !== undefined) return targetNode.scores[key];
      return 0;
    };

    let x, y, z;
    if (activeCount === 1) {
      const activeIdx = activeDimensionIndices ? activeDimensionIndices[0] : 0;
      const activeDim = dimensions[activeIdx];
      x = getScore(activeDim);
      y = 0;
      z = 0;
    } else {
      x = getScore(dimensions[0]);
      y = getScore(dimensions[1]);
      z = getScore(dimensions[2]);
    }

    return { x: scoreToPos(x), y: scoreToPos(y), z: scoreToPos(z) };
  }, [dragVisualState, nodes, dimensions, activeCount, activeDimensionIndices]);
  const mergeGhostPosition = useMemo(() => {
    if (!dragVisualState || dragVisualState.type !== 'merge') return null;
    if (hasVector3(dragVisualState.ghostPosition)) return dragVisualState.ghostPosition;
    if (!hasVector2(dragVisualState.ghostPosition)) return null;
    if (!gl?.domElement) return null;

    const rect = gl.domElement.getBoundingClientRect();
    if (!rect.width || !rect.height) return null;

    const mouse = new THREE.Vector2(
      (dragVisualState.ghostPosition.x / rect.width) * 2 - 1,
      -((dragVisualState.ghostPosition.y / rect.height) * 2 - 1)
    );

    raycaster.setFromCamera(mouse, camera);
    const viewDir = new THREE.Vector3();
    camera.getWorldDirection(viewDir);
    const planePoint = mergeTargetPosition
      ? new THREE.Vector3(mergeTargetPosition.x, mergeTargetPosition.y, mergeTargetPosition.z)
      : new THREE.Vector3(0, 0, 0);
    const plane = new THREE.Plane().setFromNormalAndCoplanarPoint(viewDir, planePoint);
    const hit = new THREE.Vector3();
    const intersected = raycaster.ray.intersectPlane(plane, hit);
    if (!intersected) return null;
    return { x: hit.x, y: hit.y, z: hit.z };
  }, [dragVisualState, gl, camera, raycaster, mergeTargetPosition]);

  useEffect(() => {
    if (activeCount < 3) {
      // Determine target face and up vector to align axes with screen
      // Requirement: First dimension (Horizontal) -> Left to Right
      //              Second dimension (Vertical) -> Bottom to Top
      let face = 0; // Default Front
      let up = new THREE.Vector3(0, 1, 0);

      if (activeCount === 2) {
        const indices = activeDimensionIndices || [0, 1];
        const [a, b] = [...indices].sort((i, j) => i - j);

        if (a === 0 && b === 1) {
          // XY: Front View (Looking at Z-)
          // X (Horizontal): Left(-X) -> Right(+X)
          // Y (Vertical): Bottom(-Y) -> Top(+Y)
          face = 0;
          up.set(0, 1, 0);
        } else if (a === 0 && b === 2) {
          // XZ: Bottom View (Looking at Y+)
          // X (Horizontal): Left(-X) -> Right(+X)
          // Z (Vertical): Bottom(-Z is Down? No, we want A(-Z) at Bottom).
          // If Up is Z+, Screen Top is Z+, Screen Bottom is Z-.
          // So A(-Z) is Bottom. B(+Z) is Top. Correct.
          face = 5;
          up.set(0, 0, 1);
        } else if (a === 1 && b === 2) {
          // YZ: Right View (Looking at X-)
          // Y (Horizontal): Left(-Y) -> Right(+Y)
          // Z (Vertical): Bottom(-Z) -> Top(+Z) (With Up=Z+)
          face = 1;
          up.set(0, 0, 1);
        }
      } else {
        // 1D: Always Front (mapped to X)
        face = 0;
        up.set(0, 1, 0);
      }

      setTargetFaceIndex(face);
      setTargetUp(up);
      setIsSnapping(true);
      setSnapped(false);
    }
  }, [activeCount, activeDimensionIndices]);

  return (
    <>
      <CameraController targetFaceIndex={targetFaceIndex} targetUp={targetUp} isSnapping={isSnapping} setSnapped={setSnapped} setIsSnapping={setIsSnapping} controlsRef={controlsRef} />
      <ambientLight intensity={0.8} />
      <pointLight position={[100, 100, 100]} intensity={1} />
      <pointLight position={[-100, -100, -100]} intensity={0.5} />

      <TrackballControls
        ref={controlsRef}
        noZoom
        noPan
        rotateSpeed={3}
        enabled={!isDraggingNode && !isSnapping && activeCount === 3}
        onStart={handleStart}
        onEnd={handleEnd}
      />

      {activeCount === 3 && <WireframeBox />}

      {/* Render ProjectedAxis based on activeCount */}
      {activeCount === 1 && <ProjectedAxis type="x" />}
      {activeCount === 2 && (
        <>
          {activeDimensionIndices.includes(0) && <ProjectedAxis type="x" />}
          {activeDimensionIndices.includes(1) && <ProjectedAxis type="y" />}
          {activeDimensionIndices.includes(2) && <ProjectedAxis type="z" />}
        </>
      )}
      {activeCount === 3 && (
        <>
          <ProjectedAxis type="x" />
          <ProjectedAxis type="y" />
          <ProjectedAxis type="z" />
        </>
      )}

      {activeCount === 1 ? (
        // For 1D, pass only the single active dimension (mapped to index 0 for correct X-axis labeling)
        <AxisLabels3D activeFace={targetFaceIndex} dims={[dimensions[activeDimensionIndices[0]], null, null]} isSnapped={isSnapped} />
      ) : (
        // For 2D/3D, pass all dims (nulls will be filtered inside AxisLabels3D component logic itself)
        <AxisLabels3D activeFace={targetFaceIndex} dims={dimensions} isSnapped={isSnapped} />
      )}

      {dragVisualState && dragVisualState.type === 'modify' && dragVisualState.ghostPosition && dragVisualState.newPosition && (
        <>
          <Line points={[
            [dragVisualState.ghostPosition.x, dragVisualState.ghostPosition.y, dragVisualState.ghostPosition.z],
            [dragVisualState.newPosition.x, dragVisualState.newPosition.y, dragVisualState.newPosition.z]
          ]} color="#9ca3af" lineWidth={2} dashed dashSize={1} gapSize={1} />
          <mesh position={[dragVisualState.ghostPosition.x, dragVisualState.ghostPosition.y, dragVisualState.ghostPosition.z]}>
            <sphereGeometry args={[3.8, 24, 24]} />
            <meshBasicMaterial color="#9ca3af" opacity={0.5} transparent />
          </mesh>
        </>
      )}
      {dragVisualState && dragVisualState.type === 'merge' && mergeGhostPosition && mergeTargetPosition && dragVisualState.targetNodeId && (
        <>
          <Line points={[
            [mergeGhostPosition.x, mergeGhostPosition.y, mergeGhostPosition.z],
            [mergeTargetPosition.x, mergeTargetPosition.y, mergeTargetPosition.z]
          ]} color="#9ca3af" lineWidth={2} dashed dashSize={1} gapSize={1} />
          <mesh position={[mergeGhostPosition.x, mergeGhostPosition.y, mergeGhostPosition.z]}>
            <sphereGeometry args={[3.8, 24, 24]} />
            <meshBasicMaterial color="#9ca3af" opacity={0.5} transparent />
          </mesh>
        </>
      )}

      {dimensions.filter(d => d !== null).length === 0 && (<Billboard position={[0, 0, 0]}><Text fontSize={6} color="#ef4444" bg="#fee2e2">Select Dimensions</Text></Billboard>)}

      {validNodes.map(node => {
        if (node.isGhost) return null;

        // Check if all required dimension scores are available
        // In 1D/2D mode, we only care about active dimensions
        const hasAllScores = dimensions.every((dimObj, idx) => {
          if (!dimObj) return true; // Inactive dimension
          // For 1D, we only care about the single active dimension
          if (activeCount === 1 && idx !== activeDimensionIndices[0]) return true;
          // For 2D, we only care about the two active dimensions
          if (activeCount === 2 && !activeDimensionIndices.includes(idx)) return true;

          const key = `${dimObj.dimensionA}-${dimObj.dimensionB}`;
          return node.scores && node.scores[key] !== undefined;
        });

        // Don't render node if scoring is incomplete
        if (!hasAllScores) return null;

        const getScore = (dimObj) => {
          if (!dimObj) return 0; // Changed from 50 to 0 (center in -50 to 50 range)
          const key = `${dimObj.dimensionA}-${dimObj.dimensionB}`;
          if (node.scores && node.scores[key] !== undefined) return node.scores[key];
          return 0; // Changed from 50 to 0
        };

        let x, y, z;
        if (activeCount === 1) {
          // 1D View: Remap the single active dimension to X axis
          // Find the active dimension
          const activeIdx = activeDimensionIndices ? activeDimensionIndices[0] : 0;
          const activeDim = dimensions[activeIdx];
          x = getScore(activeDim);
          y = 0; // Center (0 is the center in -50 to 50 range)
          z = 0; // Center
        } else {
          // 2D/3D View: Standard mapping
          x = getScore(dimensions[0]);
          y = getScore(dimensions[1]);
          z = getScore(dimensions[2]);
        }

        const overridePos = dragVisual3D && dragVisual3D.sourceId === node.id && dragVisual3D.current;
        const pos = overridePos
          ? [overridePos.x, overridePos.y, overridePos.z]
          : [scoreToPos(x), scoreToPos(y), scoreToPos(z)];

        const color = getNodeColor(node, colorMap);

        return (
          <Node3D
            key={node.id} node={node} position={pos} color={color}
            isSelected={selectedNode?.id === node.id} isHovered={hoveredNode?.id === node.id}
            onPointerOver={onNodeHover} onPointerOut={(n) => onNodeHover(null)}
            onClick={onNodeClick} onDragEnd={onNodeDragEnd}
            axisMapping={axisMapping} isSnapped={isSnapped} allNodes={validNodes} dims={dimensions} setIsDraggingNode={setIsDraggingNode}
            setDragVisual3D={setDragVisual3D} setDragVisualState={setDragVisualState} projectToFace={projectToPlane} dragVisual3D={dragVisual3D} dragVisualState={dragVisualState}
            activeCount={activeCount} // Pass activeCount to Node3D
            activeDimensionIndices={activeDimensionIndices} // Pass activeDimensionIndices to Node3D
            dragHoverTarget={dragHoverTarget}
          />
        );
      })}
    </>
  );
};

const Evaluation3D = (props) => {
  const [targetFaceIndex, setTargetFaceIndex] = useState(0);
  const colorMap = useMemo(() => ({
    root: '#4C84FF',
    simple: '#45B649',
    complex: '#FF6B6B',
  }), []);
  const activeDims = useMemo(() => {
    if (!props.selectedDimensionPairs) return [null, null, null];
    const indices = props.activeDimensionIndices || [0, 1, 2];
    return [0, 1, 2].map(i => {
      if (indices.includes(i)) return props.selectedDimensionPairs[i];
      return null;
    });
  }, [props.selectedDimensionPairs, props.activeDimensionIndices]);

  // Determine if we should show the operation status
  const showOperationStatus = props.operationStatus &&
    (props.operationStatus.toLowerCase().includes('merging') ||
      props.operationStatus.toLowerCase().includes('evaluating') ||
      props.operationStatus.toLowerCase().includes('modifying') ||
      props.isGenerating);

  return (
    <div
      data-uatrack-suppress-click="true"
      data-uatrack-suppress-hover="true"
      style={{ width: '100%', height: '100%', position: 'relative', background: '#fff', borderRadius: '8px', border: '1px solid #e5e7eb', overflow: 'hidden' }}
    >
      {/* Operation Status Banner */}
      {showOperationStatus && (
        <div style={{
          position: 'absolute',
          top: 0,
          left: 0,
          right: 0,
          zIndex: 10,
          background: 'rgba(255, 255, 255, 0.95)',
          padding: '12px',
          textAlign: 'center',
          fontSize: '1.2rem',
          fontWeight: '600',
          color: '#374151',
          borderBottom: '1px solid #e5e7eb',
          pointerEvents: 'none'
        }}>
          {props.operationStatus || 'Processing...'}
        </div>
      )}
      <Canvas shadows>
        <OrthographicCamera makeDefault zoom={4} position={[0, 0, 200]} near={0.1} far={1000} />
        <SceneContent
          nodes={props.nodes}
          dimensions={activeDims}
          activeDimensionIndices={props.activeDimensionIndices}
          onNodeDragEnd={props.onNodeDragEnd}
          selectedNode={props.selectedNode}
          onNodeClick={props.onNodeClick}
          hoveredNode={props.hoveredNode}
          onNodeHover={props.onNodeHover}
          targetFaceIndex={targetFaceIndex}
          setTargetFaceIndex={setTargetFaceIndex}
          pendingChange={props.pendingChange}
          pendingMerge={props.pendingMerge}
          colorMap={colorMap}
          dragVisualState={props.dragVisualState}
          setDragVisualState={props.setDragVisualState}
          mergeAnimationState={props.mergeAnimationState}
          onDropExternal={props.onDropExternal}
          dragHoverTarget={props.dragHoverTarget}
          onDragHover={props.onDragHover}
        />
      </Canvas>
      <div style={{ position: 'absolute', bottom: 10, left: 10, pointerEvents: 'none', background: 'rgba(255,255,255,0.8)', padding: '4px 8px', borderRadius: '4px', fontSize: '12px', color: '#666' }}>
        Drag Cube to Rotate • Drag Nodes to Edit
      </div>
    </div>
  );
};

export default Evaluation3D;
