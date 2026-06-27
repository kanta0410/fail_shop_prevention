'use client';

import { useState, useMemo, useEffect } from 'react';
import dynamic from 'next/dynamic';
import locationsData from '../data/locations.json';

const Map = dynamic(() => import('../components/Map'), { ssr: false });

type SnsStrength = 'strong' | 'weak' | 'none';

interface Location {
  id: number;
  name: string;
  lat: number;
  lng: number;
  slots: number;
  type: string;
  station_dist: number;
  office_count: number;
  competitor_count: number;
  sns_weak_score: number;
  sns_strong_score: number;
  color: string;
  reasons_weak: string[];
  reasons_strong: string[];
  features?: string[];
}

export default function Home() {
  const [selectedSnsStrength, setSelectedSnsStrength] = useState<SnsStrength>('none');
  const [selectedBusinessType, setSelectedBusinessType] = useState('cafe');
  const [selectedPurpose, setSelectedPurpose] = useState('none');

  const [appliedSnsStrength, setAppliedSnsStrength] = useState<SnsStrength>('none');
  const [appliedBusinessType, setAppliedBusinessType] = useState('cafe');
  const [appliedPurpose, setAppliedPurpose] = useState('none');

  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [hasAnalyzed, setHasAnalyzed] = useState(false);

  // 出店場所リストを状態として管理
  const [locations, setLocations] = useState<Location[]>(locationsData);

  // クライアントマウント確認用フラグ
  const [isMounted, setIsMounted] = useState(false);
  useEffect(() => {
    Promise.resolve().then(() => {
      setIsMounted(true);
    });
  }, []);

  // 初回ロード時にAPIから店舗データをロード
  useEffect(() => {
    // ローカルストレージのバックアップ読み込みを一時的に無効化（データリセット用）
    const backup = null; // localStorage.getItem('locations_backup');
    if (backup) {
      try {
        const parsed = JSON.parse(backup);
        if (Array.isArray(parsed) && parsed.length > 0) {
          // ESLint警告回避のため、非同期でstateを更新
          Promise.resolve().then(() => {
            setLocations(parsed);
          });
          
          // バックアップ内容をサーバー側の locations.json にも自動同期・保存
          fetch('/api/locations', {
            method: 'PUT',
            headers: {
              'Content-Type': 'application/json',
            },
            body: backup,
          })
            .then(res => {
              if (res.ok) console.log('Successfully auto-synced backup to server');
            })
            .catch(err => console.error('Failed to auto-sync backup to server:', err));
          
          return;
        }
      } catch (e) {
        console.error('Failed to parse locations_backup:', e);
      }
    }

    // バックアップがない場合は通常通りAPIから最新の店舗データをロード
    fetch('/api/locations')
      .then(res => res.json())
      .then(data => {
        if (Array.isArray(data)) {
          setLocations(data);
        }
      })
      .catch(err => console.error('Failed to load locations from API:', err));
  }, []);

  // 分析ボタンのハンドラ
  const handleAnalyze = () => {
    setIsAnalyzing(true);
    setHasAnalyzed(false);

    // 疑似AIローディング (1.5秒)
    setTimeout(() => {
      setAppliedBusinessType(selectedBusinessType);
      setAppliedPurpose(selectedPurpose);
      setAppliedSnsStrength(selectedSnsStrength);
      setIsAnalyzing(false);
      setHasAnalyzed(true);
    }, 1500);
  };

  // 新規出店場所の登録ハンドラ
  const handleAddLocation = (name: string, lat: number, lng: number) => {
    const nextId = locations.length > 0 ? Math.max(...locations.map(l => l.id)) + 1 : 1;
    
    // 緯度経度からそれらしいタイプを決定論的に決定
    const types = ['office', 'commercial', 'hybrid', 'public'] as const;
    const typeIndex = Math.floor(Math.abs(Math.sin(lat + lng) * 1000)) % types.length;
    const type = types[typeIndex];
    
    // 座標から決定論的にAI分析スコアを算出（リアリティのため）
    const seedWeak = Math.abs(Math.sin(lat * 12.9898 + lng * 78.233)) * 43219.1234;
    const sns_weak_score = Math.floor((seedWeak % 48) + 50); // 50〜97
    
    const seedStrong = Math.abs(Math.sin(lat * 78.233 + lng * 12.9898)) * 43219.1234;
    const sns_strong_score = Math.floor((seedStrong % 48) + 50); // 50〜97
    
    // スコアに応じて色を決定
    const color = sns_weak_score >= 75 ? 'blue' : 'red';
    
    // タイプに応じたそれらしいAI分析理由の割り当て
    const reasons_weak: string[] = [];
    const reasons_strong: string[] = [];
    
    if (type === 'office') {
      reasons_weak.push(
        "周辺オフィスの昼休み需要（ランチ難民）を確実に取り込める立地です",
        "提供スピードを重視したオペレーションにより、短時間で高い売上が期待できます",
        "SNS広告に頼らずとも、ビジネスパーソンの徒歩移動ルート上で自然と認知されます"
      );
      reasons_strong.push(
        "平日のオフィス需要に特化しているため、土日祝日の集客は弱まる傾向にあります",
        "SNSによる広域集客よりも、周辺ビルへのチラシ配りなど地域密着の認知拡大が有効です",
        "高単価な嗜好品メニューより、毎日のランチとして選びやすい価格帯のメニューが適しています"
      );
    } else if (type === 'commercial') {
      reasons_weak.push(
        "人通りが多い商業エリアで、平日・休日問わず多様な客層のトラフィックがあります",
        "周囲に飲食店が多いため競合は激しいですが、外食ニーズ自体の市場規模が最大です",
        "通行人の目を引く看板デザインや、呼び込み用のポップ設置が売上を左右します"
      );
      reasons_strong.push(
        "駅から近くSNS映えする要素と相性が良いため、インフルエンサーを通じた拡散が狙えます",
        "「この場所限定」の限定メニューなどを用意することで、遠方からの目的来店を促せます",
        "休日の若年層やファミリー層の集客力が極めて高く、高単価メニューの販売も期待できます"
      );
    } else if (type === 'hybrid') {
      reasons_weak.push(
        "オフィスワーカーと、近くのイベントスペースに集まる利用客の両方の需要を狙えます",
        "曜日や時間帯によって客層が大きく変化するため、臨機応変なメニュー変更が効果的です",
        "視認性の高い角地であり、特別な集客を行わなくても安定した客足が見込めます"
      );
      reasons_strong.push(
        "イベント開催時の人流増加とSNSプロモーションを掛け合わせることで、バズ効果を最大化できます",
        "特定の「推し活」やトレンドに合わせたコラボメニューなど、目的来店を狙う仕掛けがハマります",
        "ファンコミュニティを作りやすく、SNSのフォロワーをリピーターに育成しやすいエリアです"
      );
    } else {
      reasons_weak.push(
        "周辺に飲食店が極めて少なく、出店できればこのエリアの軽食・ランチ需要を独占できます",
        "一般の通行量は少なめであるため、地域住民や特定施設の利用者にターゲットを絞る必要があります",
        "固定客（リピーター）が定着するまでは、一時的に集客に時間を要する可能性があります"
      );
      reasons_strong.push(
        "競合が皆無であるため、SNSで「隠れ家的な人気スポット」としてブランディングが可能です",
        "「わざわざ行く価値がある」こだわりの本格メニュー（自家焙煎珈琲など）と非常に親和性が高いです",
        "競合対比でのユニークさが際立つため、SNSを活用したニッチマーケティングに最適です"
      );
    }

    const newLocation: Location = {
      id: nextId,
      name,
      lat,
      lng,
      slots: Math.floor((seedWeak % 3) + 1),
      type,
      station_dist: Math.floor((seedStrong % 350) + 50),
      office_count: Math.floor((seedWeak % 120) + 10),
      competitor_count: Math.floor((seedStrong % 100) + 5),
      sns_weak_score,
      sns_strong_score,
      color,
      reasons_weak,
      reasons_strong,
      features: ['周辺の人流密度', '交通アクセス']
    };

    const nextLocations = [...locations, newLocation];
    setLocations(nextLocations);
  };

  // 入力されたSNSの強さに応じてスコアをソート
  const rankedLocations = useMemo(() => {
    const scored = locations.map(loc => {
      let score = appliedSnsStrength === 'strong' ? loc.sns_strong_score : loc.sns_weak_score;

      // 来店目的による補正
      if (appliedPurpose === 'destination') {
        if (loc.color === 'red') score += 15;
        if (loc.type === 'public') score += 10;
      } else if (appliedPurpose === 'casual') {
        if (loc.color === 'blue') score += 15;
        if (loc.type === 'commercial' || loc.type === 'office') score += 10;
        if (loc.station_dist < 150) score += 10;
      }

      // SNS集客力による補正
      if (appliedSnsStrength === 'strong') {
        if (loc.color === 'red') score += 20;
      } else {
        if (loc.color === 'blue') score += 15;
      }

      // 業態による補正
      if (appliedBusinessType === 'bento') {
        if (loc.type === 'office') score += 20;
        if (loc.office_count > 60) score += 10;
      } else if (appliedBusinessType === 'sweets') {
        if (loc.type === 'commercial') score += 15;
        if (loc.station_dist < 150) score += 10;
      } else if (appliedBusinessType === 'specialty') {
        if (loc.type === 'public' || loc.type === 'hybrid') score += 15;
        if (loc.competitor_count < 30) score += 10;
      } else if (appliedBusinessType === 'cafe' || appliedBusinessType === 'restaurant') {
        if (loc.type === 'commercial' || loc.type === 'hybrid') score += 10;
      } else if (appliedBusinessType === 'truck') {
        if (loc.slots >= 2) score += 10;
        if (loc.type === 'office' || loc.type === 'hybrid') score += 10;
      }

      return { ...loc, computedScore: score };
    });

    // スコア順にソート
    scored.sort((a, b) => b.computedScore - a.computedScore);

    // 動的に理由（判断基準）を生成して付与
    return scored.slice(0, 3).map(loc => {
      const reasons: string[] = [];

      // 理由1: 業態と店舗タイプ（特徴量1）
      if (appliedBusinessType === 'bento') {
        reasons.push(`お弁当販売に最適な立地です。オフィスが集積するエリアであり、強力な判断基準である「${loc.features?.[0] || 'オフィス密集度'}」の恩恵を最大化して昼休みの高い需要を吸収できます。`);
      } else if (appliedBusinessType === 'specialty') {
        reasons.push(`こだわり専門店に相性の良い静かな環境です。「${loc.features?.[0] || '競合店舗数'}」が抑えられているため、競合に埋もれず独自の強いブランド価値をアピールする出店が可能です。`);
      } else if (appliedBusinessType === 'sweets') {
        reasons.push(`テイクアウトスイーツに適した動線です。主な判断基準である「${loc.features?.[0] || '歩行者通行量'}」が非常に高く、通行人の視線を引きやすいため、手軽な衝動買いを誘発できます。`);
      } else {
        reasons.push(`本業態に適合する立地特性を持っています。このエリアの主な強みである「${loc.features?.[0] || '立地ポテンシャル'}」を活かして、無駄のない店舗認知とスムーズな集客が期待できます。`);
      }

      // 理由2: 来店目的と店舗タイプ（特徴量2）
      if (appliedPurpose === 'destination') {
        reasons.push(`「わざわざ行く」目的来店に適した落ち着いたエリアです。駅近の一等地に拘らず「${loc.features?.[1] || '駅から距離'}」がある分、家賃コストを低く抑え、隠れ家的な演出でリピーターを獲得できます。`);
      } else if (appliedPurpose === 'casual') {
        reasons.push(`「フラッと立ち寄る」日常利用に絶好の環境です。主要な判断基準である「${loc.features?.[1] || '駅から距離'}」が近くアクセス性に優れるため、移動中のついで買いやフリー客を自然に呼び込めます。`);
      } else {
        reasons.push(`日常利用と週末の目的利用の双方に対応できる中立的なエリアです。バランスの取れた「${loc.features?.[1] || '地域環境'}」を背景に、安定した営業の基盤を築くことができます。`);
      }

      // 理由3: SNS集客力と立地特性
      if (appliedSnsStrength === 'strong') {
        if (loc.color === 'red') {
          reasons.push(`強固なSNS集客力を活かせるエリアです。人通りは少ないニッチな場所（注意エリア）ですが、ネット上の話題性で「知る人ぞ知る名店」としてファンを直接誘導できる好相性な立地です。`);
        } else {
          reasons.push(`強い情報発信力と好立地の相乗効果が狙えます。視認性が高くおすすめの場所（安全地帯）であるため、SNS拡散とリアルの通行客の双方からアプローチして爆発的な回転率を生み出せます。`);
        }
      } else {
        if (loc.color === 'blue') {
          reasons.push(`SNS発信に頼らなくとも安定する安全地帯です。エリア自体の自然な通行量と好立地ポテンシャルがカバーするため、店頭の工夫（看板やメニュー看板など）だけで確実な集客が望めます。`);
         } else {
          reasons.push(`SNSが弱い段階ではフリー客と地域密着のアナログアプローチが主軸となります。立地的な不利を補うため、まずは周辺 of オフィスや住民への地道なポスティングや認知作りが重要となります。`);
        }
      }

      return {
        ...loc,
        computedReasons: reasons
      };
    });
  }, [locations, appliedBusinessType, appliedPurpose, appliedSnsStrength]);

  return (
    <div className="container">
      <h1>🚚 出店場所の提案アプリ</h1>

      <div className="card">
        <h2>出店条件を入力</h2>
        <div className="form-group">
          <label>出店する業態</label>
          <select value={selectedBusinessType} onChange={(e) => setSelectedBusinessType(e.target.value)}>
            <option value="cafe">カフェ・軽食系</option>
            <option value="sweets">スイーツ・テイクアウト専門系</option>
            <option value="bento">お弁当・惣菜（ランチ特化）系</option>
            <option value="restaurant">レストラン（しっかり食事）系</option>
            <option value="specialty">こだわり専門店（自家焙煎珈琲など）系</option>
            <option value="other">その他・多国籍系</option>
          </select>
        </div>

        <div className="form-group">
          <label>主な来店目的（任意）</label>
          <select value={selectedPurpose} onChange={(e) => setSelectedPurpose(e.target.value)}>
            <option value="none">指定しない / どちらでも</option>
            <option value="casual">フラッと寄り（通りすがり）</option>
            <option value="destination">目的来店（わざわざ来る）</option>
          </select>
        </div>

        <div className="form-group">
          <label>現在のSNS集客力（任意）</label>
          <select value={selectedSnsStrength} onChange={(e) => setSelectedSnsStrength(e.target.value as SnsStrength)}>
            <option value="none">指定しない / まだない</option>
            <option value="weak">フォロワーが少ない（受動的集客メイン）</option>
            <option value="strong">フォロワーが多い（能動的集客が可能）</option>
          </select>
        </div>

        <button className="btn" onClick={handleAnalyze} disabled={isAnalyzing}>
          {isAnalyzing ? '分析中...' : 'AIでポテンシャルを分析する'}
        </button>
      </div>

      {isAnalyzing && (
        <div className="card loading">
          <div className="spinner"></div>
          <p>AIモデルが最適な立地を計算しています...</p>
        </div>
      )}

      {hasAnalyzed && !isAnalyzing && (
        <>
          <div className="card">
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '15px', flexWrap: 'wrap', gap: '10px' }}>
              <h2 style={{ margin: 0 }}> 出店ポテンシャルマップ</h2>
            </div>
            <p style={{ marginBottom: '15px', color: '#555', fontSize: '0.9rem' }}>
              ※青色は現在のあなたの条件での「安全地帯(おすすめ)」、赤色は「注意(ニッチ向け)」を示します。<br />
              <strong>💡 地図上の開いている場所をクリックすると、その場所を新しい候補地として登録できます！</strong>
            </p>
            {isMounted && <Map locations={locations} topRankedIds={rankedLocations.map(loc => loc.id)} onAddLocation={handleAddLocation} />}
          </div>

          <div className="card">
            <h2> おすすめ出店場所 トップ3</h2>
            {rankedLocations.map((loc, index) => {
              const reasons = loc.computedReasons;
              return (
                <div key={loc.id} className="ranking-item">
                  <div className="ranking-header">
                    <div className={`rank-badge rank-${index + 1}`}>{index + 1}</div>
                    <h3 style={{ margin: 0 }}>{loc.name}</h3>
                  </div>
                  <ul className="reasons-list">
                    {reasons.map((reason, rIndex) => (
                      <li key={rIndex}>{reason}</li>
                    ))}
                  </ul>
                </div>
              );
            })}
          </div>
        </>
      )}
    </div>
  );
}
