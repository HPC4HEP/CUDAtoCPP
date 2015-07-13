//----------------------------------------------------------------------//
// Source tool using libTooling.					//
//									//
// Based on tooling_sample.cpp by Eli Bendersky (eliben@gmail.com)	//
//									//
// Luca Atzori (luca.atzori@cern.ch)					//
//----------------------------------------------------------------------//

#include <string>
#include <iostream>

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

using namespace clang;
using namespace clang::ast_matchers;
using namespace clang::driver;
using namespace clang::tooling;

static llvm::cl::OptionCategory MatcherSampleCategory("Matcher Sample");


class KernelDefHandler : public MatchFinder::MatchCallback {
public:
	KernelDefHandler(Rewriter &Rewrite) : Rewrite(Rewrite) {}

	virtual void run(const MatchFinder::MatchResult &Result){
		auto f = Result.Nodes.getNodeAs<clang::Decl>("kdef");
		//const clang::Decl* f1 = Result.Nodes.getNodeAs<clang::Decl>("cex");
		Rewrite.InsertText(f->getLocStart(), "//Found a Kernel Def\n", true, true);
	}
private:
	Rewriter &Rewrite;
};

class KernelCallHandler : public MatchFinder::MatchCallback {
public:
	KernelCallHandler(Rewriter &Rewrite) : Rewrite(Rewrite) {}
	virtual void run(const MatchFinder::MatchResult &Result){
		auto kc = Result.Nodes.getNodeAs<clang::CUDAKernelCallExpr>("kcall");
		Rewrite.InsertText(kc->getLocStart(), "//Found a Kernel Call\n", true, true);
	}
private:
	Rewriter &Rewrite;
};

class MyASTConsumer : public ASTConsumer {
public:
	//ctor in which we have to add Matcher.addMatcher defs
	//always initialize handlers with rewriters
	MyASTConsumer(Rewriter &R) : KCH(R), KDH(R){

		//Trying to match Kernel Def
		//TODO: works only without macro definition
		//e.g.: NO #define __global__ __attribute__((global))
		Matcher.addMatcher(
				functionDecl(
						hasAttr(
								clang::attr::CUDAGlobal
								)
						).bind("kdef"),
				&KDH);

		//Trying to match Kernel Call
		Matcher.addMatcher(clang::ast_matchers::CUDAKernelCallExpr().bind("kcall"),
				&KCH);



	}

	// Run the matchers when we have the whole TU parsed.
	void HandleTranslationUnit(ASTContext &Context) override {
		Matcher.matchAST(Context);

	// Other technique to match Kernel Def, seems fine!
	// But the how to do rewritings?
//	    for(DeclContext::decl_iterator D = Context.getTranslationUnitDecl()->decls_begin(), DEnd = Context.getTranslationUnitDecl()->decls_end(); D!=DEnd; ++D){
//	    	FunctionDecl* FD;
//	    	//Stmt* Body;
//	    	if((FD=dyn_cast<FunctionDecl> (*D)) != 0){
//	    		if(FD->hasAttr<CUDAGlobalAttr>())
//	    			std::cout << "Matching Kernel Def!\n"
//							  << FD->getNameAsString()
//							  << "\n\n";
//	    	}
//	    }

	}

private:
	MatchFinder Matcher;
	KernelCallHandler KCH;
	KernelDefHandler KDH;
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
    return llvm::make_unique<MyASTConsumer>(TheRewriter);
  }

private:
  Rewriter TheRewriter;
};


int main(int argc, const char **argv) {
  CommonOptionsParser op(argc, argv, MatcherSampleCategory);
  ClangTool Tool(op.getCompilations(), op.getSourcePathList());

  return Tool.run(newFrontendActionFactory<MyFrontendAction>().get());
}
