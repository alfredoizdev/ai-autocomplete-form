import { NextRequest, NextResponse } from 'next/server';

// Simple in-memory storage for demo purposes
// In production, this would be stored in a database
const detectionLogs: any[] = [];

export async function POST(req: NextRequest) {
  try {
    const logs = await req.json();
    
    // Validate the logs
    if (!Array.isArray(logs)) {
      return NextResponse.json(
        { error: 'Invalid logs format' },
        { status: 400 }
      );
    }
    
    // Store logs (in production, save to database)
    detectionLogs.push(...logs);
    
    // Log to console for debugging
    console.log(`Received ${logs.length} kick detection logs`);
    logs.forEach(log => {
      console.log(`- Text: "${log.text}" | Detected: ${log.result.detected} | Confidence: ${log.result.confidence}% | Action: ${log.userAction}`);
    });
    
    return NextResponse.json({ 
      success: true, 
      received: logs.length,
      total: detectionLogs.length 
    });
  } catch (error) {
    console.error('Error processing kick detection logs:', error);
    return NextResponse.json(
      { error: 'Failed to process logs' },
      { status: 500 }
    );
  }
}

// Optional: GET endpoint to retrieve logs for analysis
export async function GET() {
  return NextResponse.json({
    logs: detectionLogs,
    count: detectionLogs.length
  });
}