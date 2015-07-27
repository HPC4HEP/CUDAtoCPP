/*
 * Matcher.cpp
 *
 *  Created on: 24/lug/2015
 *      Author: latzori
 */


#include <string>
#include <iostream>
#include <list>
#include <map>
#include <set>
#include <sstream>

#include "clang/AST/AST.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "llvm/Support/raw_ostream.h"


#include "clang/AST/Decl.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Frontend/FrontendPluginRegistry.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Lex/PPCallbacks.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Regex.h"

using namespace clang;
using namespace clang::ast_matchers;
using namespace clang::driver;
using namespace clang::tooling;
using namespace llvm::sys::path;

static llvm::cl::OptionCategory MatcherSampleCategory("Matcher Sample");

class AttributeHandler : public MatchFinder::MatchCallback {
public:
	AttributeHandler(Rewriter &Rewrite, Preprocessor* PP) : Rewrite(Rewrite), PP(PP){}

	virtual void run(const MatchFinder::MatchResult &Result){
		clang::SourceManager* const SM = Result.SourceManager;
		const clang::FunctionDecl * hostfunc = Result.Nodes.getNodeAs<clang::FunctionDecl>("hostattr");
		CUDAHostAttr* attr = hostfunc->getAttr<CUDAHostAttr>();
		SourceLocation attrloc = SM->getExpansionLoc(attr->getLocation());

	    std::string SStr;
	    llvm::raw_string_ostream S(SStr);
		hostfunc->print(S);
		std::cout << "hostfunc is " << S.str() << "\n";

		//How the hell i clean the buffer?
		attrloc.print(S,*SM);
		std::cout << "attrloc is " << S.str() << "\n";

		SourceLocation attrend = PP->getLocForEndOfToken(attrloc);
		attrend.print(S,*SM);
		std::cout << "attrend is " << S.str() << "\n";

		SourceRange sr(attrloc, attrend);
		Rewrite.ReplaceText(attrloc,
								Rewrite.getRangeSize(sr),
								"");
	}

private:
	Rewriter &Rewrite;
	Preprocessor *PP;
};

class MyASTConsumer : public ASTConsumer {
public:
	MyASTConsumer(CompilerInstance *comp, Rewriter &R) : CI(comp)  {

		Preprocessor* P = &CI->getPreprocessor();
		AttributeHandler *AH = new AttributeHandler(R, P);
		//AH(R, &CI->getPreprocessor());

		Matcher.addMatcher(
				functionDecl(
						hasAttr(
								clang::attr::CUDAHost
								)
						).bind("hostattr"),
				AH);

	}
	// Run the matchers when we have the whole TU parsed.
	void HandleTranslationUnit(ASTContext &Context) override {
		Matcher.matchAST(Context);

	}

private:
	MatchFinder Matcher;
	//AttributeHandler AH;

	//Preprocessor* PP;
	CompilerInstance *CI;

};

// For each source file provided to the tool, a new FrontendAction is created.
class MyFrontendAction : public ASTFrontendAction {
public:
  MyFrontendAction() {}
  void EndSourceFileAction() override {
    TheRewriter.getEditBuffer(TheRewriter.getSourceMgr().getMainFileID())
        .write(llvm::outs());
  }

  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI,
                                                 StringRef file) override {
    TheRewriter.setSourceMgr(CI.getSourceManager(), CI.getLangOpts());
    return llvm::make_unique<MyASTConsumer>(&CI, TheRewriter);
  }

private:
  Rewriter TheRewriter;
};

int main(int argc, const char **argv) {
  CommonOptionsParser op(argc, argv, MatcherSampleCategory);
  ClangTool Tool(op.getCompilations(), op.getSourcePathList());

  return Tool.run(newFrontendActionFactory<MyFrontendAction>().get());
}

