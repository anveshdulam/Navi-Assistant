import { NextResponse } from "next/server";

export async function GET() {
  return NextResponse.json({
    success: true,
    message: "NavAssist API ready.",
    data: { status: "ok" },
  });
}