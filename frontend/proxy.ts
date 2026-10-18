import { NextResponse } from "next/server";

// Прокси RunPod кэширует статичные страницы Next (s-maxage = 1 год),
// и после пересборки снаружи остаётся старая версия. HTML не кэшируем.
export function proxy() {
  const res = NextResponse.next();
  res.headers.set("Cache-Control", "no-store, must-revalidate");
  res.headers.set("CDN-Cache-Control", "no-store");
  return res;
}

export const config = { matcher: ["/((?!_next/static|_next/image|favicon.ico).*)"] };
