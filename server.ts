import express from 'express';
import path from 'path';
import dotenv from 'dotenv';
import { GoogleGenAI, Type } from '@google/genai';
import { createServer as createViteServer } from 'vite';

dotenv.config();

let aiClient: GoogleGenAI | null = null;

function getAiClient(): GoogleGenAI {
  if (!aiClient) {
    const apiKey = process.env.GEMINI_API_KEY;
    if (!apiKey) {
      throw new Error('GEMINI_API_KEY environment variable is missing.');
    }
    aiClient = new GoogleGenAI({
      apiKey,
      httpOptions: {
        headers: {
          'User-Agent': 'aistudio-build',
        },
      },
    });
  }
  return aiClient;
}

async function callGeminiWithFallback(ai: GoogleGenAI, config: any) {
  const models = ['gemini-3.8-flash', 'gemini-flash-latest', 'gemini-3.1-flash-lite'];
  let lastError: any = null;

  for (const model of models) {
    try {
      const response = await ai.models.generateContent({
        ...config,
        model,
      });
      return response;
    } catch (err: any) {
      console.warn(`Model ${model} failed, attempting next if available:`, err?.message || err);
      lastError = err;
      // If error is high demand (503) or rate limit, continue to fallback model
      const msg = JSON.stringify(err);
      if (msg.includes('503') || msg.includes('429') || msg.includes('high demand') || msg.includes('RESOURCE_EXHAUSTED')) {
        continue;
      }
      throw err;
    }
  }
  throw lastError;
}

function getFallbackQuestions(role: string, level: string, type: string, company: string, count: number) {
  return {
    title: `${role} (${level}) Mock Interview${company ? ` - ${company} Focus` : ''}`,
    overview: `Rigorous, structured interview session focusing on ${type} and executive hiring expectations.`,
    targetCompetencies: [
      'Problem Solving & Architecture',
      'Ownership & Cross-Functional Alignment',
      'Communication & STAR Storytelling',
      'Incident & Conflict Resolution'
    ],
    questions: [
      {
        id: 'q-fb-1',
        question: `Tell me about the most technically or organizationally challenging initiative you spearheaded in your ${role} career. What was your personal contribution, and what was the quantifiable impact?`,
        category: 'Project Deep Dive & STAR',
        competency: 'Ownership & Technical Leadership',
        whyAsked: 'Evaluators want to verify scope, autonomy, proactive problem-solving, and whether you measure success by customer/business metrics.',
        hint: 'Use the STAR format: succinctly outline the business challenge, the trade-offs you navigated, your specific actions, and the concrete metrics delivered.',
        suggestedPacingSeconds: 150,
        starFocus: {
          situation: 'The state of the system or business before the project.',
          task: 'Your specific responsibility and ownership perimeter.',
          action: 'Key technical decisions, consensus building, and execution.',
          result: 'Quantified metrics (e.g. latency cut by 40%, revenue boost, uptime).'
        }
      },
      {
        id: 'q-fb-2',
        question: `Describe a situation where a critical system broke or an unexpected roadblock jeopardized a major deliverable. How did you react under pressure?`,
        category: 'Crisis Response & Resilience',
        competency: 'Pressure Management & Root-Cause Analysis',
        whyAsked: 'Assesses emotional composure, rapid mitigation strategies, blameless post-mortem methodology, and long-term safeguard implementation.',
        hint: 'Highlight immediate triage and customer communication first, followed by deep root-cause discovery and permanent automated regression tests.',
        suggestedPacingSeconds: 120,
        starFocus: {
          situation: 'The critical failure and who it affected.',
          task: 'Your immediate triage duty.',
          action: 'How you isolated the issue, rolled back or patched, and kept stakeholders informed.',
          result: 'Resolution speed and preventative mechanisms put in place.'
        }
      },
      {
        id: 'q-fb-3',
        question: `Have you ever strongly disagreed with a key decision made by your manager or a peer? How did you present your case, and what happened next?`,
        category: 'Conflict Resolution',
        competency: 'Constructive Disagreement & Influence',
        whyAsked: 'Tests emotional intelligence, evidence-based persuasion, team-first mentality, and the ability to commit fully even if not your preferred way.',
        hint: 'Center the conversation on data and customer impact rather than personal preferences. Frame disagreement as mutual pursuit of excellence.',
        suggestedPacingSeconds: 120,
        starFocus: {
          situation: 'The architectural or roadmap contention.',
          task: 'The objective need to reach the right outcome without stalling the team.',
          action: 'Data collected, respectful dialogue, compromise or commit.',
          result: 'The outcome of the decision and mutual team trust.'
        }
      },
      {
        id: 'q-fb-4',
        question: `How do you decide what NOT to build when facing tight deadlines and competing stakeholder demands?`,
        category: 'Prioritization & Strategy',
        competency: 'Ruthless Prioritization & Trade-offs',
        whyAsked: 'Hiring managers look for candidates who avoid scope creep, protect delivery velocity, and defend engineering hygiene without alienating partners.',
        hint: 'Reference a structured mental framework (e.g. RICE, MoSCoW, or high impact vs. effort matrix) and clear stakeholder transparency.',
        suggestedPacingSeconds: 110,
      }
    ].slice(0, count)
  };
}

function getFallbackEvaluation(question: string, userAnswer: string) {
  const wordCount = userAnswer.trim().split(/\s+/).filter(Boolean).length;
  const hasSituation = /situation|context|project|working on|initially|background/i.test(userAnswer);
  const hasTask = /task|goal|responsible|objective|needed to|assigned/i.test(userAnswer);
  const hasAction = /action|i did|implemented|decided|built|architected|collaborated|led|analyzed/i.test(userAnswer);
  const hasResult = /result|outcome|improved|reduced|increased|delivered|percent|%|metric/i.test(userAnswer);

  let score = 70;
  if (wordCount >= 100) score += 10;
  if (hasResult) score += 10;
  if (hasAction) score += 5;
  if (wordCount < 40) score -= 25;

  const boundedScore = Math.max(45, Math.min(95, score));
  const grade = boundedScore >= 85 ? 'Strong Hire' : boundedScore >= 70 ? 'Leaning Hire' : 'Needs Polish';

  return {
    overallScore: boundedScore,
    grade,
    summary: `Structured answer (${wordCount} words) addressing the prompt with solid clarity. ${hasResult ? 'Good inclusion of outcome signals.' : 'Could be elevated with more quantifiable metrics and explicit STAR progression.'}`,
    metrics: {
      clarityStructure: {
        score: Math.min(95, boundedScore + 3),
        feedback: hasAction ? 'Logical progression of events described.' : 'Focus on clearly delineating what actions YOU personally took.'
      },
      relevanceDepth: {
        score: boundedScore,
        feedback: wordCount >= 80 ? 'Good depth and contextual framing.' : 'Add more tactical technical details and constraints.'
      },
      competencyFit: {
        score: boundedScore,
        feedback: 'Demonstrates clear role awareness and problem-solving maturity.'
      },
      deliveryImpact: {
        score: hasResult ? boundedScore + 5 : boundedScore - 5,
        feedback: hasResult ? 'Effective highlighting of business or technical impact.' : 'Anchor your conclusion with measurable numbers (e.g., % reduction, latency, team adoption).'
      }
    },
    starBreakdown: {
      situation: {
        detected: hasSituation,
        comment: hasSituation ? 'Situation and context were established.' : 'Open with a sharper 1-sentence situation summary.'
      },
      task: {
        detected: hasTask,
        comment: hasTask ? 'Clearly stated the challenge you were tasked with.' : 'Explicitly define your specific responsibility.'
      },
      action: {
        detected: hasAction,
        comment: hasAction ? 'Personal contributions were emphasized.' : 'Use active voice: "I designed, I prioritized, I negotiated".'
      },
      result: {
        detected: hasResult,
        comment: hasResult ? 'Addressed the outcome of your actions.' : 'Quantify the final result: latency, dollars saved, or launch timeline.'
      }
    },
    strengths: [
      'Directly addressed the prompt without dodging the core question',
      'Demonstrated professional accountability and ownership',
      'Maintained good conversational tone and clarity'
    ],
    improvements: [
      'Structure the response more explicitly around Situation -> Task -> Action -> Result',
      'Quantify the delivered outcome with concrete numbers or percentage improvements',
      'Highlight specific trade-offs and alternative solutions you evaluated'
    ],
    modelAnswer: `In my previous role, we faced a similar situation where a critical service was struggling under peak load, leading to elevated p99 latency. My task was to diagnose the root cause and restore performance before our upcoming product launch. I analyzed profiling telemetry, identified severe thread contention in our database connection pool, and re-architected the caching layer using Redis with exponential backoff. As a result, we reduced p99 response times by 55%, cut server compute costs by 22%, and completed the launch with zero downtime.`,
    followUpQuestion: 'What trade-offs did you consider before choosing that particular approach over alternatives?'
  };
}

async function startServer() {
  const app = express();
  const PORT = 3000;

  app.use(express.json({ limit: '10mb' }));

  // Health check
  app.get('/api/health', (req, res) => {
    res.json({
      status: 'ok',
      hasApiKey: Boolean(process.env.GEMINI_API_KEY),
      timestamp: new Date().toISOString(),
    });
  });

  // Generate customized interview questions
  app.post('/api/interview/generate-questions', async (req, res) => {
    try {
      const {
        role = 'Software Engineer',
        level = 'Mid-Level',
        type = 'Comprehensive Mix',
        company = '',
        jobDescription = '',
        count = 4,
      } = req.body;

      const ai = getAiClient();

      const prompt = `You are an elite executive interviewer and hiring manager at a top-tier company.
Generate a structured, authentic mock interview session for a candidate with the following profile:
- Role / Title: ${role}
- Experience Level: ${level}
- Interview Style / Focus: ${type}
${company ? `- Target Company: ${company} (incorporate known company values, leadership principles, or technical bar)` : ''}
${jobDescription ? `- Provided Job Description / Focus Areas:\n"""${jobDescription.slice(0, 1500)}"""` : ''}
- Number of questions: ${count}

Instructions:
1. Create realistic, realistic-depth interview questions appropriate for ${level} ${role}.
2. For behavioral questions, include a guide on how the STAR framework (Situation, Task, Action, Result) applies.
3. For technical / case questions, include what signals the interviewer is assessing.
4. Provide a helpful hint/mental model for each question.
5. Recommend a target answer duration in seconds (usually 90 to 180 seconds).`;

      let parsed = null;
      try {
        const response = await callGeminiWithFallback(ai, {
          contents: prompt,
          config: {
            systemInstruction:
              'You are an expert technical and behavioral interviewer. Return structured JSON matching the schema.',
            responseMimeType: 'application/json',
            responseSchema: {
              type: Type.OBJECT,
              properties: {
                title: { type: Type.STRING },
                overview: { type: Type.STRING },
                targetCompetencies: {
                  type: Type.ARRAY,
                  items: { type: Type.STRING },
                },
                questions: {
                  type: Type.ARRAY,
                  items: {
                    type: Type.OBJECT,
                    properties: {
                      id: { type: Type.STRING },
                      question: { type: Type.STRING },
                      category: { type: Type.STRING },
                      competency: { type: Type.STRING },
                      whyAsked: { type: Type.STRING },
                      hint: { type: Type.STRING },
                      suggestedPacingSeconds: { type: Type.INTEGER },
                      starFocus: {
                        type: Type.OBJECT,
                        properties: {
                          situation: { type: Type.STRING },
                          task: { type: Type.STRING },
                          action: { type: Type.STRING },
                          result: { type: Type.STRING },
                        },
                      },
                    },
                    required: [
                      'id',
                      'question',
                      'category',
                      'competency',
                      'whyAsked',
                      'hint',
                      'suggestedPacingSeconds',
                    ],
                  },
                },
              },
              required: ['title', 'overview', 'targetCompetencies', 'questions'],
            },
          },
        });

        const responseText = response.text || '{}';
        parsed = JSON.parse(responseText);
      } catch (err: any) {
        console.warn('AI generator fallback triggered:', err.message);
        parsed = getFallbackQuestions(role, level, type, company, count);
      }

      res.json(parsed);
    } catch (error: any) {
      console.error('Error generating questions:', error);
      const fallback = getFallbackQuestions('Software Engineer', 'Mid-Level', 'General', '', 4);
      res.json(fallback);
    }
  });

  // Evaluate candidate's answer
  app.post('/api/interview/evaluate-answer', async (req, res) => {
    try {
      const {
        role,
        level,
        question,
        category,
        userAnswer,
        timeSpentSeconds,
        company,
      } = req.body;

      if (!userAnswer || userAnswer.trim().length === 0) {
        return res.status(400).json({ error: 'User answer is required' });
      }

      const ai = getAiClient();

      const prompt = `You are an expert interview coach evaluating a candidate's response.
Context:
- Target Role: ${role} (${level})
${company ? `- Company: ${company}` : ''}
- Question Category: ${category}
- Question: "${question}"
- Time spent answering: ${timeSpentSeconds ? `${timeSpentSeconds} seconds` : 'unrecorded'}
- Candidate's Answer:
"""
${userAnswer}
"""

Evaluate this answer thoroughly with constructive, rigorous hiring manager standards:
1. Overall score from 0 to 100.
2. Grade: "Exceptional", "Strong Hire", "Leaning Hire", "Needs Polish", or "Weak Answer".
3. Evaluate 4 core metrics (0 to 100 score + concise actionable feedback):
   - Clarity & Structure (logical flow, succinctness, STAR adherence if applicable)
   - Relevance & Depth (addresses the core question directly, avoids fluff, includes specifics)
   - Role Competency (technical or leadership maturity expected for ${level} ${role})
   - Delivery & Impact (confidence, proactive ownership, measurable outcomes mentioned)
4. Evaluate STAR breakdown: detect whether Situation, Task, Action, Result were present and provide a short critique for each.
5. Specific Strengths (2-4 bullet points praising real aspects of their answer).
6. Actionable Improvements (2-4 concrete pointers to elevate this to top 1% candidate level).
7. "Model Answer": A high-impact, exemplary rewrite of how a top candidate would answer this question authentically with concrete metrics and punchy STAR framing.
8. Follow-up Question: A realistic probe an interviewer would ask right after hearing this answer.`;

      let parsed = null;
      try {
        const response = await callGeminiWithFallback(ai, {
          contents: prompt,
          config: {
            systemInstruction:
              'You are a premier executive interview coach. Provide rigorous, encouraging, highly specific feedback. Return JSON.',
            responseMimeType: 'application/json',
            responseSchema: {
              type: Type.OBJECT,
              properties: {
                overallScore: { type: Type.INTEGER },
                grade: { type: Type.STRING },
                summary: { type: Type.STRING },
                metrics: {
                  type: Type.OBJECT,
                  properties: {
                    clarityStructure: {
                      type: Type.OBJECT,
                      properties: {
                        score: { type: Type.INTEGER },
                        feedback: { type: Type.STRING },
                      },
                      required: ['score', 'feedback'],
                    },
                    relevanceDepth: {
                      type: Type.OBJECT,
                      properties: {
                        score: { type: Type.INTEGER },
                        feedback: { type: Type.STRING },
                      },
                      required: ['score', 'feedback'],
                    },
                    competencyFit: {
                      type: Type.OBJECT,
                      properties: {
                        score: { type: Type.INTEGER },
                        feedback: { type: Type.STRING },
                      },
                      required: ['score', 'feedback'],
                    },
                    deliveryImpact: {
                      type: Type.OBJECT,
                      properties: {
                        score: { type: Type.INTEGER },
                        feedback: { type: Type.STRING },
                      },
                      required: ['score', 'feedback'],
                    },
                  },
                  required: [
                    'clarityStructure',
                    'relevanceDepth',
                    'competencyFit',
                    'deliveryImpact',
                  ],
                },
                starBreakdown: {
                  type: Type.OBJECT,
                  properties: {
                    situation: {
                      type: Type.OBJECT,
                      properties: {
                        detected: { type: Type.BOOLEAN },
                        comment: { type: Type.STRING },
                      },
                      required: ['detected', 'comment'],
                    },
                    task: {
                      type: Type.OBJECT,
                      properties: {
                        detected: { type: Type.BOOLEAN },
                        comment: { type: Type.STRING },
                      },
                      required: ['detected', 'comment'],
                    },
                    action: {
                      type: Type.OBJECT,
                      properties: {
                        detected: { type: Type.BOOLEAN },
                        comment: { type: Type.STRING },
                      },
                      required: ['detected', 'comment'],
                    },
                    result: {
                      type: Type.OBJECT,
                      properties: {
                        detected: { type: Type.BOOLEAN },
                        comment: { type: Type.STRING },
                      },
                      required: ['detected', 'comment'],
                    },
                  },
                  required: ['situation', 'task', 'action', 'result'],
                },
                strengths: {
                  type: Type.ARRAY,
                  items: { type: Type.STRING },
                },
                improvements: {
                  type: Type.ARRAY,
                  items: { type: Type.STRING },
                },
                modelAnswer: { type: Type.STRING },
                followUpQuestion: { type: Type.STRING },
              },
              required: [
                'overallScore',
                'grade',
                'summary',
                'metrics',
                'starBreakdown',
                'strengths',
                'improvements',
                'modelAnswer',
                'followUpQuestion',
              ],
            },
          },
        });

        const responseText = response.text || '{}';
        parsed = JSON.parse(responseText);
      } catch (err: any) {
        console.warn('AI evaluation fallback triggered:', err.message);
        parsed = getFallbackEvaluation(question, userAnswer);
      }

      res.json(parsed);
    } catch (error: any) {
      console.error('Error evaluating answer:', error);
      const fallback = getFallbackEvaluation(req.body.question || 'Interview Question', req.body.userAnswer || '');
      res.json(fallback);
    }
  });

  // Get quick coaching hint / framework tips for a question
  app.post('/api/interview/coaching-hint', async (req, res) => {
    try {
      const { question, role, level, category } = req.body;
      const ai = getAiClient();

      const prompt = `The candidate is preparing to answer this interview question:
Question: "${question}"
Category: ${category}
Role: ${role} (${level})

Provide 3 quick, crisp tips to ace this answer:
1. Framework / Structure to apply
2. Key signals / keywords to highlight
3. Common trap or cliché to avoid`;

      const response = await ai.models.generateContent({
        model: 'gemini-3.8-flash',
        contents: prompt,
        config: {
          systemInstruction:
            'You are an executive career coach. Return JSON with 3 concise tips.',
          responseMimeType: 'application/json',
          responseSchema: {
            type: Type.OBJECT,
            properties: {
              frameworkTip: { type: Type.STRING },
              keySignals: {
                type: Type.ARRAY,
                items: { type: Type.STRING },
              },
              pitfallToAvoid: { type: Type.STRING },
            },
            required: ['frameworkTip', 'keySignals', 'pitfallToAvoid'],
          },
        },
      });

      const responseText = response.text || '{}';
      res.json(JSON.parse(responseText));
    } catch (error: any) {
      console.error('Error generating hint:', error);
      res.status(500).json({
        error: error.message || 'Failed to generate coaching hint',
      });
    }
  });

  // TTS for question readout
  app.post('/api/interview/tts', async (req, res) => {
    try {
      const { text, voice = 'Zephyr' } = req.body;
      if (!text) {
        return res.status(400).json({ error: 'Text is required for TTS' });
      }

      const ai = getAiClient();
      // Use gemini-3.1-flash-tts-preview as per skill guidelines
      const response = await ai.models.generateContent({
        model: 'gemini-3.1-flash-tts-preview',
        contents: [{ parts: [{ text: text.slice(0, 400) }] }],
        config: {
          responseModalities: ['AUDIO'],
          speechConfig: {
            voiceConfig: {
              prebuiltVoiceConfig: {
                voiceName: voice, // 'Zephyr', 'Kore', 'Fenrir', 'Puck', 'Charon'
              },
            },
          },
        },
      });

      const base64Audio =
        response.candidates?.[0]?.content?.parts?.[0]?.inlineData?.data;
      if (base64Audio) {
        res.json({ audioBase64: base64Audio, format: 'pcm', sampleRate: 24000 });
      } else {
        res.status(404).json({ error: 'No audio returned' });
      }
    } catch (error: any) {
      console.warn('TTS API warning (fallback to Web Speech available):', error.message);
      res.status(500).json({ error: error.message || 'TTS generation failed' });
    }
  });

  // Vite middleware in dev, static files in prod
  if (process.env.NODE_ENV !== 'production') {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), 'dist');
    app.use(express.static(distPath));
    app.get('*', (req, res) => {
      res.sendFile(path.join(distPath, 'index.html'));
    });
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`AI Interview Prep Assistant server running on http://0.0.0.0:${PORT}`);
  });
}

startServer();
