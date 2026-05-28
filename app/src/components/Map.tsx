'use client';

import { useState } from 'react';
import { MapContainer, TileLayer, CircleMarker, Circle, Popup, Tooltip, useMapEvents } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';

// Fix for default marker icon missing in Leaflet with React
import L from 'leaflet';
// eslint-disable-next-line @typescript-eslint/no-explicit-any
delete (L.Icon.Default.prototype as any)._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://unpkg.com/leaflet@1.7.1/dist/images/marker-icon-2x.png',
  iconUrl: 'https://unpkg.com/leaflet@1.7.1/dist/images/marker-icon.png',
  shadowUrl: 'https://unpkg.com/leaflet@1.7.1/dist/images/marker-shadow.png',
});

interface Location {
  id: number;
  name: string;
  lat: number;
  lng: number;
  color: string;
  features?: string[];
}

interface MapProps {
  locations: Location[];
  onAddLocation?: (name: string, lat: number, lng: number) => void;
}

function MapEventsHelper({ onMapClick }: { onMapClick: (lat: number, lng: number) => void }) {
  useMapEvents({
    click(e) {
      onMapClick(e.latlng.lat, e.latlng.lng);
    }
  });
  return null;
}

export default function Map({ locations, onAddLocation }: MapProps) {
  // 名古屋の中心付近
  const center: [number, number] = [35.1706, 136.9034];
  
  // 新規追加用の一時マーカー
  const [tempMarker, setTempMarker] = useState<{ lat: number; lng: number } | null>(null);
  const [newName, setNewName] = useState('');

  const handleMapClick = (lat: number, lng: number) => {
    setTempMarker({ lat, lng });
    setNewName('');
  };

  const handleRegister = () => {
    if (!tempMarker) return;
    const name = newName.trim() || `新規候補地 (${tempMarker.lat.toFixed(4)}, ${tempMarker.lng.toFixed(4)})`;
    if (onAddLocation) {
      onAddLocation(name, tempMarker.lat, tempMarker.lng);
    }
    setTempMarker(null);
    setNewName('');
  };

  const handleCancel = () => {
    setTempMarker(null);
    setNewName('');
  };

  return (
    <div style={{ height: '400px', width: '100%', marginBottom: '20px', borderRadius: '8px', overflow: 'hidden', position: 'relative' }}>
      <MapContainer center={center} zoom={14} style={{ height: '100%', width: '100%' }}>
        {/* 薄めのベースマップ（CartoDB Positron）を使用してヒートマップ風の演出をサポート */}
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'
          url="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png"
        />
        
        {/* レイヤーを重ねて疑似的なグラデーションヒートマップを作る */}
        {locations.flatMap(loc => {
          if (typeof loc.lat !== 'number' || typeof loc.lng !== 'number' || isNaN(loc.lat)) return [];
          const baseColor = loc.color === 'blue' ? '#3498db' : '#e74c3c';
          return [
            <Circle key={`grad-1-${loc.id}`} center={[loc.lat, loc.lng]} radius={400} pathOptions={{ fillColor: baseColor, fillOpacity: 0.15, stroke: false, interactive: false }} />,
            <Circle key={`grad-2-${loc.id}`} center={[loc.lat, loc.lng]} radius={800} pathOptions={{ fillColor: baseColor, fillOpacity: 0.1, stroke: false, interactive: false }} />,
            <Circle key={`grad-3-${loc.id}`} center={[loc.lat, loc.lng]} radius={1200} pathOptions={{ fillColor: baseColor, fillOpacity: 0.05, stroke: false, interactive: false }} />,
            <Circle key={`grad-4-${loc.id}`} center={[loc.lat, loc.lng]} radius={1600} pathOptions={{ fillColor: baseColor, fillOpacity: 0.03, stroke: false, interactive: false }} />
          ];
        })}

        {/* 地図クリックイベントのヘルパー */}
        <MapEventsHelper onMapClick={handleMapClick} />
        
        {locations
          .filter((loc) => loc && typeof loc.lat === 'number' && typeof loc.lng === 'number' && !isNaN(loc.lat) && !isNaN(loc.lng))
          .map((loc) => (
            <CircleMarker
              key={loc.id}
              center={[loc.lat, loc.lng]}
              radius={12}
              pathOptions={{ 
                fillColor: loc.color === 'blue' ? '#3498db' : '#e74c3c', 
                color: 'white',
                weight: 2,
                fillOpacity: 0.8
              }}
            >
              <Tooltip permanent direction="top" opacity={0.9} offset={[0, -10]}>
                <span style={{ fontSize: '16px' }}>★</span>
              </Tooltip>
              <Popup>
                <strong>{loc.name}</strong><br />
                {loc.features && loc.features.length > 0 && (
                  <div style={{ margin: '8px 0', fontSize: '0.9em', color: '#555' }}>
                    <strong>判断基準:</strong>
                    <ul style={{ margin: '4px 0 0 20px', padding: 0 }}>
                      {loc.features.map((f, i) => <li key={i}>{f}</li>)}
                    </ul>
                  </div>
                )}
                ポテンシャル: {loc.color === 'blue' ? '高 (おすすめ)' : '注意 (ニッチ向け)'}
              </Popup>
            </CircleMarker>
          ))}

        {/* 新規登録用の一時ピン */}
        {tempMarker && (
          <CircleMarker
            center={[tempMarker.lat, tempMarker.lng]}
            radius={14}
            pathOptions={{
              fillColor: '#f1c40f', // 目立つ黄色
              color: '#d35400',
              weight: 3,
              fillOpacity: 0.9,
              dashArray: '5, 5' // 点線にして新規追加中であることを表現
            }}
          >
            <Popup autoClose={false} closeOnClick={false}>
              <div style={{ minWidth: '180px', padding: '5px 0' }}>
                <strong style={{ display: 'block', marginBottom: '8px', fontSize: '0.95rem' }}>📍 新しい出店場所を追加</strong>
                <div style={{ marginBottom: '8px' }}>
                  <label style={{ fontSize: '0.8rem', color: '#555', display: 'block', marginBottom: '4px' }}>場所の名前（任意）</label>
                  <input
                    type="text"
                    value={newName}
                    onChange={(e) => setNewName(e.target.value)}
                    placeholder="例: 名古屋栄ビル前"
                    style={{
                      width: '100%',
                      padding: '6px',
                      fontSize: '0.85rem',
                      border: '1px solid #ccc',
                      borderRadius: '4px',
                      boxSizing: 'border-box'
                    }}
                    autoFocus
                  />
                </div>
                <div style={{ display: 'flex', gap: '8px', marginTop: '10px' }}>
                  <button
                    onClick={handleRegister}
                    style={{
                      flex: 1,
                      padding: '6px',
                      backgroundColor: '#2ecc71',
                      color: 'white',
                      border: 'none',
                      borderRadius: '4px',
                      cursor: 'pointer',
                      fontSize: '0.8rem',
                      fontWeight: 'bold'
                    }}
                  >
                    登録する
                  </button>
                  <button
                    onClick={handleCancel}
                    style={{
                      padding: '6px 10px',
                      backgroundColor: '#e74c3c',
                      color: 'white',
                      border: 'none',
                      borderRadius: '4px',
                      cursor: 'pointer',
                      fontSize: '0.8rem'
                    }}
                  >
                    キャンセル
                  </button>
                </div>
              </div>
            </Popup>
          </CircleMarker>
        )}
      </MapContainer>
    </div>
  );
}
