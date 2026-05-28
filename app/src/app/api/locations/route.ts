import { NextResponse } from 'next/server';
import fs from 'fs/promises';
import path from 'path';

// JSONファイルへの絶対パス
const filePath = path.join(process.cwd(), 'src', 'data', 'locations.json');

export async function GET() {
  try {
    const fileData = await fs.readFile(filePath, 'utf-8');
    const locations = JSON.parse(fileData);
    return NextResponse.json(locations);
  } catch (error) {
    console.error('Failed to read locations:', error);
    return NextResponse.json({ error: 'Failed to read locations data' }, { status: 500 });
  }
}

export async function POST(request: Request) {
  try {
    const newLocation = await request.json();
    const fileData = await fs.readFile(filePath, 'utf-8');
    const locations = JSON.parse(fileData);

    // バリデーション
    if (!newLocation || typeof newLocation.id !== 'number') {
      return NextResponse.json({ error: 'Invalid location data' }, { status: 400 });
    }

    locations.push(newLocation);

    // インデント2文字で整形して書き込み
    await fs.writeFile(filePath, JSON.stringify(locations, null, 2), 'utf-8');

    return NextResponse.json(locations);
  } catch (error) {
    console.error('Failed to save location:', error);
    return NextResponse.json({ error: 'Failed to save location data' }, { status: 500 });
  }
}

export async function PUT(request: Request) {
  try {
    const updatedLocations = await request.json();

    if (!Array.isArray(updatedLocations)) {
      return NextResponse.json({ error: 'Invalid data format' }, { status: 400 });
    }

    // ファイルに上書き
    await fs.writeFile(filePath, JSON.stringify(updatedLocations, null, 2), 'utf-8');

    return NextResponse.json(updatedLocations);
  } catch (error) {
    console.error('Failed to save locations:', error);
    return NextResponse.json({ error: 'Failed to save data' }, { status: 500 });
  }
}
