"use client";

import { Canvas } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import type { SegmentationArtifact } from "@/lib/types";

function PointCloud({ users }: { users: SegmentationArtifact["users"] }) {
  const positions = new Float32Array(users.flatMap((user) => [((user.avgSteps - 7500) / 9000) * 2.7, ((user.avgCalories - 800) / 700) * 2.7, ((user.avgHeartRate - 110) / 45) * 2.7]));
  const palette = [[0.09, 0.27, 0.64], [0.89, 0.29, 0.23], [0.09, 0.09, 0.09], [0.42, 0.45, 0.5], [0.7, 0.32, 0.03]];
  const colors = new Float32Array(users.flatMap((user) => palette[user.cluster % palette.length]));
  return <points><bufferGeometry><bufferAttribute attach="attributes-position" args={[positions, 3]} count={users.length} array={positions} itemSize={3} /><bufferAttribute attach="attributes-color" args={[colors, 3]} count={users.length} array={colors} itemSize={3} /></bufferGeometry><pointsMaterial size={0.035} vertexColors transparent opacity={0.72} /></points>;
}

export function ClusterScene({ users }: { users: SegmentationArtifact["users"] }) {
  return <div className="h-[460px] w-full bg-[#ebe9e2]"><Canvas camera={{ position: [3.5, 2.8, 4.5], fov: 48 }} dpr={[1, 1.5]}><ambientLight intensity={1.4} /><PointCloud users={users} /><gridHelper args={[6, 12, "#b7b5ae", "#d9d7d0"]} rotation={[0, 0, 0]} /><OrbitControls enablePan={false} minDistance={2.5} maxDistance={8} /></Canvas></div>;
}